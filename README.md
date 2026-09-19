# MCT Bench

Sony Model Compression Toolkit（MCT）で公開の学習済みCNNにPTQを適用し、**FP32・W8A8・W4A8の精度、実ファイル容量、CPU推論時間と差分を1枚の表にするCLI**です。Markdown、同じ列構成のCSV、再現情報付きJSON、実行可能なONNXを保存します。

同梱：[10,000画像での実測表](examples/full/comparison.md) / [CSV](examples/full/comparison.csv) / [検証記録](VALIDATION.md)。自動テスト91件を実行済みです。

## 最初に知っておくこと

- **実測するもの**：エクスポートしたONNXのaccuracy、ファイル全体のbytes、ONNX Runtime CPUの推論時間。改善しなかった場合もそのまま出力します。
- **ビット幅の意味**：`W4A8`は重み4bit・活性値8bit。対象となる6個のConvと1個のLinearの量子化器を実際に調べ、指定と異なれば停止します。バイアスはFP32です。
- **保存形式の意味**：MCTの`FAKELY_QUANT`標準ONNXでは、量子化済み重みの値はfloat32で保存され、活性値はQuantizeLinear/DequantizeLinearで表現されます。INT4/INT8を詰め込んだファイル容量や、低ビット専用カーネルの速度を示すベンチマークではありません。
- **測定条件**：全モデルで`CPUExecutionProvider`、`ORT_DISABLE_ALL`を使用。ONNX Runtimeの最適化による量子化境界の丸め差を抑えるため、エクスポートしたグラフを最適化せずに測ります。実機や製品向けに最適化した推論速度とは異なります。
- **対象素材**：Fashion-MNIST（MIT）と`tsilva/fashionmnist-classifier-cnn`の公開学習済みONNX（モデルカードでMIT宣言）。素材の一覧・根拠・表示義務は[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)にあります。

## セットアップ

Python **3.12**を使用してください。検証環境はmacOS / Apple Silicon / Python 3.12.14。Linux x86_64 CPU向けの依存指定とCIも用意していますが、同梱の実測はmacOSで実行したものです。Windows、Intel Mac、GPU実行は検証対象外です。

このREADMEがあるディレクトリで実行します。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt`はMCT 2.6.0、mct-quantizers 1.7.0、PyTorch 2.6.0、ONNX 1.17.0、ONNX Runtime 1.21.0を含む直接・間接依存を固定しています。LinuxではPyTorch公式CPU wheel indexを使用します。OS間でwheelのバイナリは異なります。

## おすすめの実行

```bash
python -m mct_bench run --output runs/full
```

初回に約35.5 MBの公開素材を取得し、SHA-256・ファイル長を照合します。学習や微調整は行いません。既存の出力ディレクトリは上書きしません。

既定の条件：

| 項目 | 設定 |
| --- | --- |
| 比較列 | FP32、W8A8、W4A8 |
| キャリブレーション | 公式train splitから512画像、batch 32 |
| 精度評価 | 公式test splitの全10,000画像、batch 128 |
| 前処理 | float32 NCHW、`(pixel / 255 - 0.2860) / 0.3530` |
| 閾値探索 | MSE、weight bias correction有効 |
| 乱数seed | 42 |
| 推論時間 | batch 1、CPU intra-op 1 thread、inter-op 1 thread |
| ウォームアップ | 各モデル20回、計測対象外 |
| 計測 | 各モデル100回、毎回モデル順をランダム化 |

端末のstdoutには表だけを出し、進捗はstderrに出します。

```bash
python -m mct_bench run --output runs/full-2 > table.md 2> run.log
```

### 小さく動作確認

```bash
python -m mct_bench run --output runs/smoke \
  --calibration-samples 64 --eval-samples 256 --warmup 5 --repeats 20
```

少数画像での結果は動作確認です。全テストデータでの精度として扱わないでください。表に評価画像数と推論batch sizeも含めます。

### ビット幅・設定を横に並べる

```bash
python -m mct_bench run --bits 8 4 --output runs/bits
python -m mct_bench run --config configs/comparison.json --output runs/settings
```

後者は通常の2設定に加え、4bitのbias correction無効と、no-clippingによる閾値設定を比較します。すべて元の学習済みモデルから独立に量子化します。

```json
{
  "variants": [
    {"name": "w8a8", "weights_bits": 8},
    {"name": "w4a8", "weights_bits": 4},
    {"name": "w4a8-no-bias-correction", "weights_bits": 4, "bias_correction": false},
    {"name": "w4a8-no-clipping", "weights_bits": 4, "error_method": "no_clipping"}
  ]
}
```

設定項目は`name`、`weights_bits`（4/8）、`activation_bits`（8）、`error_method`（`mse`/`no_clipping`）、`bias_correction`（boolean）です。未対応の項目・値、名前の重複はエラーにします。**W4A4には対応しません**。固定したPyTorchの標準ONNX exporterが4bit活性値の範囲をサポートしないためです。[PyTorchの該当実装](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/onnx/symbolic_opset13.py)

### オフライン再実行

```bash
python -m mct_bench prepare --cache .cache/mct-bench
python -m mct_bench run --offline --cache .cache/mct-bench --output runs/offline
python -m mct_bench licenses
```

キャッシュが欠けていればオフライン実行を停止します。破損したファイルも停止し、黙って差し替えません。エラーメッセージで示された破損ファイルを削除して`prepare`を再実行してください。

## 成果物と表の読み方

```text
runs/full/
├── comparison.md             # 設定を横に並べた1枚の表
├── comparison.csv            # 同じ表をスプレッドシート向けに保存
├── results.json              # 丸め前の値、全計測サンプル、環境、量子化器監査
├── sample-indices.json       # train/testそれぞれの使用画像index
├── run-status.json           # complete / failed
├── assets_manifest.json      # 元素材の固定revision・URL・SHA-256
├── THIRD_PARTY_LICENSES.md
├── licenses/                 # 素材のライセンス・モデルカード
└── models/
    ├── fp32.onnx
    ├── w8a8.onnx
    └── w4a8.onnx
```

- Accuracy：`正解数 / 評価画像数`。10クラス分類なのでtop-1 accuracyを採用しています。AUCは本アダプターの出力に含めません。
- Accuracy delta：`100 × (量子化後accuracy − FP32 accuracy)`、単位はpercentage points（pp）。
- Size delta：`100 × (量子化後ONNX bytes / FP32 ONNX bytes − 1)`。ヘッダー・定数を含む自己完結ONNX全体を測ります。理論的な低ビット容量は代入しません。
- Latency delta：`100 × (量子化後median / FP32 median − 1)`。負なら短縮、正なら増加です。
- Speedup：`FP32 median / 量子化後median`。1未満なら遅くなっています。
- 時間は1回のbatch推論のmsです。データ取得、前処理、モデル読み込み、PTQ、export、ウォームアップは含みません。median・p90・mean・標準偏差と全サンプルを保存します。
- 各計測回で全モデルに同じ準備済み入力batchを渡します。最大16個のbatchを循環し、モデル順はseed付きで毎回入れ替えます。
- `results.json`にはCPU型番、OS、実際の依存バージョン、thread数、計測前後のload average、`--note`で指定した作業状況も記録します。温度・電源設定・他プロセスは制御しないため、別マシンで同じ時間になる保証はありません。

## 再現・検証の仕組み

1. 公開元のrevisionとSHA-256を固定し、ONNX・IDXデータを検証。
2. 安全に読めるONNXの重み定数を、ローカル定義の固定PyTorch CNNに読み込み。pickle checkpointやダウンロードしたPythonは実行しません。
3. train splitだけからseed付きでキャリブレーション画像を抽出。test splitの選択indexも保存し、全設定で同じ画像を使用。
4. 元ONNX→PyTorchとFP32再exportの出力を最大32テスト画像で照合（`atol=rtol=1e-4`）。この検査は校正値の探索には使用しません。
5. MCTでPTQし、実際の量子化器のビット幅を監査。標準ONNXの自己完結ファイルとしてexportし、ONNX checkerとORTで読み込み検証。
6. PTQ PyTorch→ONNXの出力を同じ最大32画像で確認。許容差は**実際の最終出力量子化刻み1つ＋1e-5**、相対許容差は0。丸めが量子化境界をまたぐことがあるためFP32と同じ許容差にはしていません。実際の最大差・許容差・予測一致率を保存します。閾値を超えれば成功の表を出さず停止します。
7. 以後のaccuracyと時間は、両方ともエクスポートされたONNXを同じORT設定で実行して計測。

seed固定は素材・入力選択と量子化の再現に使います。ライブラリ・OS・CPUが異なる場合の完全なbit一致は保証しません。再評価時は`results.json`の条件と同じコマンドを指定してください。

途中で失敗すると終了コード1（割り込みは130）を返し、`run-status.json`と`error.log`を残します。成功を装う空の表は生成しません。

## 自動テスト

```bash
# 単体テストと、実MCTを使う小さな統合テスト。公開素材の取得は不要。
python -m pytest -m 'not assets' -q

# 公開素材と4種類のPTQ設定を使い、CLI全体を検証。
python -m mct_bench prepare --cache .cache/mct-bench
MCT_BENCH_CACHE="$PWD/.cache/mct-bench" python -m pytest -q
```

ハッシュ不一致・IDX破損、設定エラー、元モデルの保持、重みの量子化刻み、指定ビット幅、標準ONNX、batch可変性、推論結果、端数batch、計測回数・順番・ウォームアップ除外、差分の符号、CSV/Markdown/JSONの整合、実素材でのCLI完走を検査します。統合テスト内の小さな乱数モデルは検査用fixtureで、ベンチマーク結果には使用しません。

GitHub Actionsは`.github/workflows/tests.yml`にあります。CIは公開時のpushで自動実行される設定で、結果はActionsタブに残ります。同梱の実行記録は[VALIDATION.md](VALIDATION.md)を参照してください。

## 設計と拡張点

| モジュール | 責務 |
| --- | --- |
| `assets.py` / `assets_manifest.json` | 公開素材取得、ハッシュ、IDX、正規化 |
| `model.py` | 公開ONNXの固定グラフを読み込むモデルアダプター |
| `config.py` | JSON設定と対応範囲の検査 |
| `quantization.py` | MCT TPC、PTQ、実ビット監査、ONNX export |
| `runtime.py` | 全モデル共通のORT設定、推論、export照合 |
| `measurement.py` | accuracyと公平な時間計測 |
| `reporting.py` | 1枚の比較表と生データ保存 |
| `cli.py` | 分割選択、実行、証跡の取りまとめ |

新しいモデル／データには、ライセンスを明記した別のasset manifestとアダプター、前処理・分割・export一致テストを追加する設計です。任意のモデルを自動変換できる汎用コンバーターではありません。設定違いはJSONだけで増やせます。

この実装の標準形式は**ONNX**です。TFLite経路、INT4 packed storageへの変換、エッジデバイス固有の変換・書き込みは含みません。

## ライセンスと一次資料

このCLIのコードは閲覧用に公開しています。読む・手元で動かす以外の利用は許可制です（[LICENSE](LICENSE)）。データ・モデルは各公開元のライセンスに従います。実行結果のONNXを再配布する際にも、同じrunに保存した`licenses/`と`THIRD_PARTY_LICENSES.md`を添えてください。

- [MCT 2.6.0](https://github.com/SonySemiconductorSolutions/mct-model-optimization/tree/v2.6.0)
- [MCT ONNX export API実装](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/v2.6.0/model_compression_toolkit/exporter/model_exporter/pytorch/pytorch_export_facade.py)
- [MCTの量子化済み重みの保存実装](https://github.com/SonySemiconductorSolutions/mct-model-optimization/blob/v2.6.0/model_compression_toolkit/exporter/model_exporter/pytorch/base_pytorch_exporter.py)
- [公開モデルカード（固定revision）](https://huggingface.co/tsilva/fashionmnist-classifier-cnn/blob/8b068e4063f2b03ea91651b0fa5877550616d6ae/README.md)
- [Fashion-MNIST（固定revision）](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141)
