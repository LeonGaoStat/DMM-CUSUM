# DMM-CUSUM 仿真与定位诊断代码

本包为待上传代码，提供论文中本方法的仿真与定位诊断代码。实证代码单独保存在“实证代码”文件夹，不包含在本包中。

## 代码与大致运行时间

以下时间按本机完整实验规模给出，仅供安排运行；成员变化与鲁棒性为估计时间。

| 入口 | 用途及对应图表 | 大致时间 |
|---|---|---|
| `run_proposed.py` | 表2，本方法IC仿真 | 约1.5小时 |
| `run_proposed.py` | 表3，本方法活动变化OC | 约1小时 |
| `run_proposed.py` | 表4，本方法成员变化OC，读取原固定控制限 | 约40分钟（估计） |
| `run_proposed.py --include-robustness` | 补充表S2，鲁棒性 | 约3小时（估计） |
| `run_table5_proposed.py` | 表5，本方法定位与类型诊断 | 约3分钟 |

## 仿真运行

仿真使用Python 3.12.2，依赖见`requirements-current-mac.txt`。定位诊断使用Python 3.9.6，依赖见`localization/requirements.txt`。

```bash
python run_proposed.py --help
python run_proposed.py --list
python run_proposed.py --execute
```

无参数和`--list`仅显示清单。显式`--execute`运行IC、活动变化和成员变化，共8步；鲁棒性需另加`--include-robustness`，共9步。输出分别保存到`rerun_proposed/core/`和`rerun_proposed/with_robustness/`。已有运行使用`--resume`，保持原配置和选项。

成员变化读取`simulation/Table6_membership_shift_final/calibration/`中三个情景的原固定控制限。`proposed_workflow_commands.json`记录完整参数；主稿表号与底层脚本旧文件名中的表号不完全相同，以本说明为准。

主要实现文件：`simulation/code/run_arl0_nested_20261102.py`负责IC，`run_table5_proposed_batch.py`负责活动变化；`simulation/Table6_membership_shift_final/code/run_table6_proposed_batch.py`负责成员变化；`simulation/code/run_robustness_nested_20261104.py`负责鲁棒性。其他基础模块是这些程序的依赖。

## 定位诊断运行

```bash
python run_table5_proposed.py --help
python run_table5_proposed.py --execute --out localization/results_proposed_01
```

定位入口仅运行固定的论文情景，不执行情景搜索。
`copied_source_manifest.json`记录源码与必要输入的版本；
`membership_saved_inputs_manifest.json`记录原固定控制限输入。
共用基础模块保留原函数，推荐通过本包入口运行。
