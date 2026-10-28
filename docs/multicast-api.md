# 组播转写管理接口说明

multi-stream-asr 的组播管理：通过 REST 接口增删组播转写任务，每开一路自动从会话池分配一个 lane（默认上限 8 路），无需挂 WebSocket；需要看实时转写结果时再连 WS「收听」。任务地址持久化到磁盘，**服务重启后自动恢复**。

服务地址：`http://<板子IP>:8000`

## 概念

- **组播任务（job）**：一路 UDP 组播音频 → ASR 转写，占一个 lane。任务由服务端持有，与任何 WebSocket 连接无关——页面关掉任务照跑。
- **收听（listener）**：WS 连 `/ws/multicast` 挂到某个任务上接收转写事件，可挂多个、可随时断开，不影响任务本身。
- **持久化清单**：`output/multicast_jobs.json`（相对服务 WorkingDirectory，板端为 `/opt/multi-stream-asr1/output/multicast_jobs.json`）。开启任务时写入，手动停止时移除；服务正常关停不清空，启动时按清单自动拉起全部任务。

## REST 接口

### 1. 查询任务列表

```
GET /v1/multicast
```

返回：

```json
{
  "jobs": [
    {
      "group": "224.2.110.220",
      "port": 6020,
      "lane_id": 0,
      "listeners": 1,
      "finished": false,
      "packets_received": 51000,
      "frames_emitted": 10623,
      "heartbeats_sent": 30
    }
  ],
  "max_jobs": 8
}
```

字段：`lane_id` 分配到的转写通道；`listeners` 当前收听者数量；`packets_received`/`frames_emitted` 收包/出帧计数（为 0 说明音源没推流）；`heartbeats_sent` CMD_B 心跳数。

### 2. 开启一路组播

```
POST /v1/multicast/start
Content-Type: application/json
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `group` | string | （必填） | 组播地址，必须在 224.0.0.0/4 内 |
| `port` | int | 6020 | 组播端口 |
| `local_ip` | string | 0.0.0.0 | 加入组播的本机网卡 IP，0.0.0.0 表示所有网卡都加入 |
| `packet_bytes` | int | 1280 | UDP 包大小（sm-sdk relay 默认 1280） |
| `use_full_packet` | bool | false | 整包都是 PCM（false 时按 sm-sdk 包格式解析） |
| `heartbeat_interval` | float | 5.0 | CMD_B 心跳间隔秒数，0 关闭。硬件组播源靠心跳唤醒推流 |

成功返回 `{"ok": true, "group": ..., "port": ..., "lane_id": 0, "note": "..."}`。

错误：
- `400` 地址非法 / 不是组播地址 / 端口越界
- `400 组播路数已达上限 8，请先停止一路再新增` —— lane 满了
- 同一 `group:port` 重复开启不报错，直接复用已有任务（幂等）

### 3. 停止一路组播

```
POST /v1/multicast/stop
Content-Type: application/json

{"group": "224.2.110.220", "port": 6020}
```

停止任务、释放 lane、从持久化清单移除。任务不存在返回 `404`。

### 4. 修改已有任务的组播地址

```
POST /v1/multicast/modify
Content-Type: application/json

{"old_group": "224.2.110.220", "old_port": 6020, "new_group": "224.2.110.221", "new_port": 6021}
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `old_group` / `old_port` | string / int | 6020 | 要修改的现有任务地址 |
| `new_group` | string | （必填） | 新组播地址，必须在 224.0.0.0/4 内 |
| `new_port` | int | 沿用旧端口 | 新端口 |

语义：沿用旧任务的接收参数（local_ip / packet_bytes / use_full_packet / heartbeat_interval），停旧开新，持久化清单同步更新。

错误：旧任务不存在 `404`；新地址非法 `400`；lane 满了 `400`。

注意：旧任务上的收听连接（WS listener）不会迁移，客户端需用新地址重新收听。

## WS 收听转写结果

```
ws://<板子IP>:8000/ws/multicast?group=224.2.110.220&port=6020&wakeup=false
```

连接后发送 `{"type": "start"}` 挂到任务上；任务不存在时会先创建（等价于调一次 start）。再发 `{"type": "finish"}` 停止任务并释放 lane。

查询参数：`group`、`port`、`local_ip`、`packet_bytes`、`use_full_packet`、`heartbeat_interval`、`wakeup`（唤醒门控，纯转写场景建议 `false`）。

事件流与 `/ws/stream` 相同：`transcript`（partial/final）、`speaker_remap`、`status` 等。

**WS 断开不会停止任务**；「停止收听」只是摘掉自己这个 listener。

> WS 消息级切换组播地址（`set_multicast_group`）见 [组播地址修改接口说明.md](组播地址修改接口说明.md)。

## 持久化与重启恢复

| 时机 | 行为 |
|---|---|
| 开启任务 | 地址追加进 `output/multicast_jobs.json` |
| 手动停止任务 | 地址从清单移除 |
| 服务正常关停（stop_all） | **不清空清单** |
| 服务启动 | 模型加载完后按清单逐路拉起，自动分配 lane；坏条目（非组播地址、端口越界）跳过并记日志 |

恢复时用的启动参数固定为：`local_ip=0.0.0.0`、`packet_bytes=1280`、`use_full_packet=false`、`heartbeat_interval=5.0`。

注意：
- 恢复的是**任务**，不是收听连接——重启后页面需重新点「收听」。
- 清单文件手工编辑也可以，但要在服务停止时改，运行中改会被覆盖。

## 测试页

`tests/speaker_test.html`（浏览器直接打开，Ctrl+F5 强刷避免缓存）：

- 「组播管理」栏：输入组地址+端口 → **开启组播**；地址簿下拉会记住输过的地址（浏览器 localStorage）
- 列表每路：**收听**（新开 WS 看该路实时转写，可同时收听多路）/ **修改**（改组播地址，原收听断开）/ **停止**（彻底关停释放 lane）
- 转写行带彩色地址标签区分来自哪一路

## 部署注意事项

- 路数上限 = 服务启动参数 `--lanes`（板端 8）。要改上限先改 systemd 的 `--lanes`。
- **跨网段必须配组播路由**：板子改 IP 后如果收不到包/心跳发不出去（日志 `Network is unreachable`），检查路由表有没有 `224.0.0.0/4` 和默认网关。用项目根的 `change-ip.sh` 改 IP 会自动带上组播路由。
- 排查收包：`tcpdump -i end1 -nn host <组播地址>`；查任务计数：`GET /v1/multicast` 看 `packets_received` 是否增长。
