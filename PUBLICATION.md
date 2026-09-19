# MCT Bench — Zenn完成稿とキャラクター付きプレゼン

## 開くもの

- **[プレゼンを開く](presentation/presentation.html)**：10枚のHTMLスライド。小さなロボットが吹き出しと日本語の音声で説明します。
- **[完成記事を読む](article-preview.html)**：ブラウザー向けの読みやすいプレビュー。
- [Zenn用Markdown原稿](articles/mct-ptq-onnx-benchmark.md)：front matter付きの完成稿。
- [Zenn Webエディター貼付用の本文](zenn-editor-body.md)：タイトル・front matterを除いた同一本文。
- [実測表](evidence/comparison.md) / [元の数値JSON](evidence/results.json) / [素材のライセンス一覧](evidence/THIRD_PARTY_LICENSES.md)。

記事の題名：**Sony MCTで量子化を実測するCLIを作った。4bitでも容量が減らなかった理由**

## プレゼンの操作

`presentation/presentation.html`をChrome、Edge、Safariなどのブラウザーで開いてください。

| 操作 | 内容 |
| --- | --- |
| 説明を聞く | 今のページを読み上げる。再度押すと停止 |
| 通しで再生 | 今のページから説明し、終わると次へ進む。最後のページで終了 |
| ← / → | 前後のページへ移動。再生中の音声は停止 |
| 目次 | 10枚の任意のページへ移動 |
| 音声・操作 | 日本語音声・速度を変更、全画面、全10枚の印刷 |
| Home / End | 最初・最後のページへ移動 |
| Space / Esc | 説明の再生・停止 / 停止 |

図・文字・ロボット画像・JavaScriptはHTML内に含まれ、スライドの表示に外部ライブラリやサーバーは不要です。同梱記事や証拠ファイルへのリンクも使う場合は、フォルダー一式を保ってください。

**音声はブラウザー／OSの日本語読み上げ機能を使います。** 日本語音声がない環境でも吹き出しは読めます。音声の種類によってネット接続が必要です。録音済みの音声ファイルを同梱したものではありません。OSで動きを減らす設定をすると、キャラクターのアニメーションも止まります。

台本：[narration.md](presentation/narration.md) / [音声・出典の編集用JSON](presentation/narration.json)。キャラクターの画像は幾何学図形だけで組んだ自作SVG（[制作記録](presentation/assets/PROVENANCE.md)）で、外部素材は使っていません。架空の案内役で、Sonyの公式キャラクターではありません。

台本・出典の編集後は`python3 presentation/build.py`でHTMLへ埋め込み直せます。Python標準ライブラリのみを使います。見出しやレイアウトはHTMLを直接編集してください。

## 完成状態と検証

- 記事は完成稿として保存。Zenn 版は同じ本文です。
- GitHubは[MCT Benchリポジトリ](https://github.com/hocky0301/mct-bench)。
- 全10枚を5つの画面サイズでチェックし、横方向のはみ出し・デスクトップでの内容のはみ出し・JavaScriptエラーは0件。全10枚をPCで目視確認しました。
- Chrome for Testingで日本語音声のstart/endイベントと停止を実確認。通し再生・最終ページ終了・ページ移動でのキャンセル・音声なしの動作はテスト用音声で検証。音質を聴き比べた評価は行っていません。
- 印刷は10ページに分かれ、各ページの末尾注記まで含まれることを確認しました。
- 記事はZenn公式Markdownレンダラーで変換し、5表・8見出し・12コードブロック、貼付用本文との一致を確認。ローカル記事プレビューのMermaidはコード表示です。
- 数値は既存の本測定JSONと照合。今回、機械学習ベンチマークを再実行したものではありません。91件のPythonテストの成功は記録済みの検証結果です。GitHub ActionsのCIは公開時のpushで実行されます。

詳細：[資料の検証記録](validation/presentation.json) / [記事の検証記録](validation/article.json) / [元ベンチマークの記録](evidence/BENCHMARK_VALIDATION.md)。

## 結果を読むときの要点

W4A8は重み4bit・活性値8bitです。今回のFAKELY_QUANT形式は量子化した重みの値をfloat32で保存するため、ONNX容量は+0.18%、accuracyは−0.54 ppでした。CPUの時間は負荷のある共有ホストでの参考値で、packed INT4形式や低ビット専用カーネル、エッジ実機の性能を示していません。

実装の再現手順は記事とリポジトリREADMEにあります。記事のコマンドは、実測に対応するリリースタグ `v0.1.0` を指定します。
