# 2D 艺术二维码 Agent

把照片、插画或品牌标志融进可扫码的二维码。上传一张原图，再提供完整网址或原有二维码，工作台会生成艺术码和物料预览。确认读码通过的一稿后，可以下载无损 PNG 或包含打样文件的 ZIP。

适合制作展签、货架卡、台牌、贴纸和门贴。生成与读码均在本机进行，工作台无需模型服务或 API 密钥。

## 在 macOS 运行工作台

需要 Python 3.11、Xcode Command Line Tools 提供的 `swiftc`，以及 Poppler 提供的 `pdftoppm`。如果已安装 Homebrew，可用 `brew install poppler` 安装 Poppler。

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

打开 [本地工作台](http://127.0.0.1:8785)。端口被占用时可用 `python app.py --port 8786`。

1. 上传 PNG、JPG 或 WebP 原图，单张最高 12 MB。填入完整 HTTP(S) 网址，或上传要沿用的原二维码。网址里的查询参数和片段会原样保留。
2. 选物料版式并填写毫米宽高。可添加标题、品牌标志或现场照片，也可以圈定原图里希望保留的区域。
3. 生成两稿，对照原图、平稿和整图读码记录。确认通过的一稿后，下载艺术码 PNG 或整套 ZIP。

适用的物料 ZIP 包含平稿、按尺寸生成的 RGB 打样 PDF、场景预览、备用码、原图和读码记录。输出保存在本地的 `jobs/`。桌面读码使用 macOS Vision 与 OpenCV。纸张、贴膜和玻璃上的效果需以实物打样和扫码为准。项目支持固定网址与 RGB 打样。

## 本地验证对话交付

`migration/openagent-1.0.4/` 保存对话工具、Python 入口和工作台覆盖文件。它复用相同的生成、查码和确认下载流程。可在临时目录组装后，通过本地适配器验证出件。

先按上面步骤安装依赖。在仓库根目录运行以下命令，并把 `/absolute/path/to/your-image.png` 换成自己的测试图。

```sh
runtime=$(mktemp -d)
git archive HEAD | tar -xf - -C "$runtime"
rsync -a migration/openagent-1.0.4/ "$runtime/"
cd "$runtime"

python openagent_adapter.py preview \
  --image /absolute/path/to/your-image.png \
  --url 'https://example.org/exhibit?edition=2#label' \
  --layout label --width 100 --height 90 --single-preview
```

预览返回 JSON，其中有任务 ID、原图与码面对照图、候选名称和读码结果。查看 `preview` 指向的对照图，选 `selectable` 和 `exact_match` 都为 `true` 的候选。将下面的 `JOB_ID` 和 `CANDIDATE` 换成预览中的 `job` 和候选键，再确认打包。

```sh
python openagent_adapter.py deliver --job JOB_ID --candidate CANDIDATE --visual-confirmed
```

`native-webchat.js` 和 `native_turn.py` 是 OpenAgent / OpenClaw 的宿主接入源码，需要宿主提供会话与附件。覆盖层可通过 `ARTQR_OPENCV_ONLY=1` 使用单独的 OpenCV 读码，也可通过 `ARTQR_FONT` 指定可读取的中文字体。独立工作台的启动与使用无需宿主。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `app.py`、`templates/`、`static/` | 本地工作台、选稿确认与下载 |
| `art.py`、`illustration.py`、`source_fusion.py` | 图案融合、二维码绘制和物料排版 |
| `scan.py`、`research/二维码双向延展/workflows/` | 图像对照、二维码模块优化和整图读码 |
| `migration/` | 对话适配器、宿主接入及共用 HTTP 辅助函数 |

本项目以 MIT 许可发布。
