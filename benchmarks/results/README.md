# Recorded benchmark logs

These are the terminal transcripts from the valid baseline (`R000`) and
released champion (`R107`) evaluations. They are plain-text logs, not the
generated Markdown reports.

| Revision | Tool-call quality | Throughput grid | Geometric mean `tg t/s` |
| --- | ---: | ---: | ---: |
| `R000` | 97/100, 15/15 complete | 18/18 complete | 51.91233349565498 |
| `R107` | 97/100, 15/15 complete | 18/18 complete | 84.89430291791149 |

R107 improves the fixed-grid geometric mean by 63.53397584220921% over R000
while retaining the same tool-call score and completion rate.

The archived files are:

- `r000/tool-calls.txt`
- `r000/throughput.txt`
- `r107/tool-calls.txt`
- `r107/throughput.txt`

For readability, ANSI color escape sequences, machine-local paths, and run
timestamps were removed. Benchmark values and relevant terminal output were
otherwise left unchanged. The exact workload and server configuration are
documented in [`../profiles/r107-sm120.md`](../profiles/r107-sm120.md).

Original source checksums before the readability-only cleanup:

```text
fb14a7847a079d591296eae4b557b1c50ba2f0921b69bccf3a00f6003dec6c68  R000/standard.log
ed17bda2a7bc1a1ee84af49c713b41191af6f8191bcb08ec2d890d30571bed0e  R000/throughput.log
34c9c5da80dd8067b38b1d276351093965fb3b60ced38874f813b026ebc7a973  R107/tools-command.log
3a584325ffcae59997ed6816113b2e82e8a3a90e2fbb5dbcc0b65345df06cc6b  R107/throughput-command.log
```
