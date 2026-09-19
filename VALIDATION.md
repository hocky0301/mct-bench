# 実行・検証記録

実施日：2026-09-17（日本時間）。ローカルのApple M3 Pro / macOS arm64 / Python 3.12.14で実行しました。

## 検証結果

| 検査 | 結果 | 証跡 |
| --- | --- | --- |
| 固定依存の整合性 | `pip check`成功。`requirements.txt`のdry-runで追加・変更なし | `requirements.txt`、実測JSONの依存一覧 |
| 単体・MCT統合・公開素材CLIテスト | **91 passed**、24.12秒 | [テストログ](validation/tests.log)、[JUnit XML](validation/tests.xml) |
| 初回の新規ダウンロード | 7ファイル、35,474,653 bytes、全ハッシュ一致 | [取得検証](validation/prepare.json) |
| オフライン素材準備 | 終了コード0、オンライン時と同一結果 | 同上 |
| 公開素材による4種類のPTQ設定 | MSEのW8A8/W4A8、bias correction無効、no-clippingがCLI完走 | `tests/test_e2e.py`とJUnit XML |
| 推奨設定の全件評価 | 512画像で校正、10,000画像で各モデル評価、終了コード0 | [実測表](examples/full/comparison.md)、[全測定データ](examples/full/results.json) |
| 標準形式 | FP32/W8A8/W4A8の自己完結ONNXを保存。ONNX checker・ORTで検証 | [モデルと結果](examples/full/) |

テストの14 warningsは依存ライブラリmatplotlib/pyparsingの非推奨API警告です。GitHub ActionsのCIは、公開時のpushで自動実行される設定です（同じワークフローは公開前の検証中にmacOS・Linuxで成功していますが、公開リポジトリでの実行結果はActionsタブを参照してください）。Linuxでのインストール・推論もこの検証記録の対象外です。

## 全件評価の要約

| 指標 | FP32 | W8A8 | W4A8 |
| --- | ---: | ---: | ---: |
| Accuracy | 94.76% | 94.76% | 94.22% |
| FP32からのaccuracy差 | — | 0.00 pp | −0.54 pp |
| ONNX bytes | 4,590,949 | 4,599,177 | 4,599,177 |
| 容量差 | — | +0.18% | +0.18% |
| 推論時間median、batch 1 | 3.992 ms | 4.043 ms | 4.046 ms |
| 推論時間差 | — | +1.27% | +1.35% |

**この時間は負荷のある共有ホストで測った参考値です。** 計測前のload average（1分/5分/15分）は8.09 / 6.17 / 4.74。電源・温度・他プロセスを隔離していません。100回の計測の標準偏差は約0.11–0.17 msあり、小さな時間差から有意な性能差は主張しません。全サンプルとp90はJSONと実測表に残しています。

全モデルでCPU 1 thread、`ORT_DISABLE_ALL`、20回のウォームアップを使用。標準fake-quant ONNXでfloat32保存した重みを測るため、この結果はINT4/INT8を詰め込んだ容量や整数専用カーネルの速度を示しません。今回の保存形式で容量が増えたこともそのまま報告しています。

実行時間帯：2026-09-17 09:34:14–09:36:51 UTC（2026-09-17 18:34:14–18:36:51 JST）。元素材のキャッシュ場所を除く再現コマンド：

```bash
python -m mct_bench prepare --cache .cache/mct-bench
python -m mct_bench run --offline --cache .cache/mct-bench --output runs/reproduce \
  --note 'Local shared host; other user workloads may be active. No host isolation or power/thermal control.'
```

## 出力の一致確認

全件評価とは別に、export照合には選択したtest splitの先頭32画像を使用しました。

- 元の公開ONNX→PyTorchとFP32再export：最大logit差`8.5831e-06`、予測一致率100%。
- W8A8/W4A8のPyTorch→ONNX：最大logit差`0.0625`、最終出力量子化刻み`0.0625`、予測一致率100%。明記した許容差`0.06251`以内。
- MCTの各PTQモデルに6個のConvと1個のLinearの重み量子化器を確認。W8A8は全て8bit、W4A8は全て4bit。

この検査は全入力における完全一致の証明ではありません。accuracyの表は、照合用32画像ではなく、実際のONNXによる10,000画像全件の結果です。

## 配布物の検証

`MANIFEST.sha256`は配布物の各ファイル（manifest自身を除く）のSHA-256です。ZIP作成後に再展開し、CRC・安全な相対パス・全ファイルのハッシュと、3つのONNXの実測JSONに記録された容量・ハッシュを照合します。配布用ZIPの検査結果はZIPと並ぶ`mct-bench-package-verification.json`に保存します。

公開ソース版として保存する際に、`results.json`の`command`欄と検証ログのローカル絶対パス・PC名を匿名化しました。ベンチマーク本体・測定値・テスト結果・ONNXファイルは変更していません。
