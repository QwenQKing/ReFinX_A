# DSH dependency

The agent harness uses `@deepseek-ai/dsh` version `0.2.0-rc.2`, maintained by the public [DeepSeek Harness project](https://github.com/deepseek-ai/deepseek-harness). The submission includes an installer rather than a bundled third-party runtime.

Run the installer from the repository root with Node.js 25.9.0 and npm available:

```bash
bash vendor/dsh/install.sh "$HOME/.refinx-node/global"
export PATH="$HOME/.refinx-node/global/bin:$PATH"
dsh --version
```

The optional first argument selects the installation prefix. Installation requires network access to npm and pins the DSH version. Transitive dependencies are resolved by npm and can differ from the experimental environment; `packages.tsv` preserves that environment's package-version and license inventory.

`LICENSE` preserves the DSH MIT notice. Dependencies downloaded by npm retain their respective licenses.
