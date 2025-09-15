# Customer-contact

## 概要

本プロジェクトは、顧客からの問い合わせ対応を効率化するための社内向けカスタマーサポート支援ツールです。社内に蓄積された PDF や Word、CSV 等のドキュメントを活用し、問い合わせ内容に応じた情報検索や自動応答を実現します。

## 主な機能

- 社内ドキュメント（PDF, Word, CSV 等）を横断的に検索
- 顧客情報や問い合わせ履歴の参照
- サービス・商品情報の自動抽出
- Slack 等外部サービスとの連携

## ディレクトリ構成

```
├── components.py         # UIや主要機能のコンポーネント
├── constants.py          # 定数定義
├── initialize.py         # 初期化処理
├── main.py               # エントリーポイント
├── requirements_mac.txt  # 必要パッケージ
├── utils.py              # ユーティリティ関数
├── data/                 # 各種ドキュメント・データ
│   ├── rag/              # 社内ナレッジ（PDF, Word等）
│   └── slack/            # CSVデータ
├── env/                  # 仮想環境
├── images/               # 画像素材
├── logs/                 # ログ
└── ...
```

## セットアップ方法

1. Python 3.11 系の仮想環境を作成し、`env/`を有効化
2. 依存パッケージをインストール
   ```sh
   pip install -r requirements_mac.txt
   ```
3. `main.py`を実行
   ```sh
   python main.py
   ```

## 使用技術

- Python 3.11
- Streamlit（UI）
- LangChain, OpenAI API（自然言語処理/検索）
- PyMuPDF, docx2txt（ドキュメント解析）
