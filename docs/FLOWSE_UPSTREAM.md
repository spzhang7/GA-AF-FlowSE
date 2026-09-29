# FlowSE 上游来源与本地适配

本项目把 FlowSE 作为独立的 baseline 层，源码已经放在仓库顶层
`flowse/`。`rl/` 只包含我们自己的 AF/GA-AF/OGAF 和适配代码。

```powershell
python scripts/fetch_flowse.py --verify-only
```

Linux/macOS：

```bash
python scripts/fetch_flowse.py --verify-only
```

当前固定上游版本：

```text
repository: https://github.com/Honee-W/FlowSE.git
commit:     cfb81f171d804689bf7607afd0d12a6ee89de547
```

上游来源记录在 `flowse/UPSTREAM_SOURCE.json`，兼容 patch 位于
`flowse/patches/0001-ga-af-flowse-compatibility.patch`。维护脚本可以：

1. 根据固定 commit 获取选定的上游文件；
2. 应用本项目的兼容 patch；
3. 检查源码文件是否完整。

当前仓库提交的是已经应用 patch 的 `flowse/` 源码。`--verify-only` 可以检查：

```powershell
python scripts/fetch_flowse.py --verify-only
```

本项目已取得使用和再分发 FlowSE 代码所需的授权。FlowSE 仍作为第三方基线单独声明，不纳入 GA-AF 自有代码的版权范围；正式仓库中应在 `THIRD_PARTY_NOTICES.md` 写明授权信息、原作者和论文引用。模型权重仍按各自来源和发布条件单独处理。

引用 FlowSE：

```bibtex
@misc{wang2025flowseefficienthighqualityspeech,
  title={FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching},
  author={Ziqian Wang and Zikai Liu and Xinfa Zhu and Yike Zhu and Mingshuai Liu and Jun Chen and Longshuai Xiao and Chao Weng and Lei Xie},
  year={2025},
  eprint={2505.19476},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2505.19476}
}
```

