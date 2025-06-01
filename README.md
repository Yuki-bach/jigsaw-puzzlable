# jigsaw-puzzlable

ジグソーパズルの白ピース自動マッチングシステム

## 概要

jigsaw-puzzlableは、1000ピースのジグソーパズルから残された約200個の白いピース（内部ピースのみ）を自動的に解析し、確実に繋がるピースのグループを発見するツールです。

コンピュータビジョン技術を使用してピースの輪郭を抽出し、凸凹の形状を比較することで高精度なマッチングを実現しています。

## 特徴

- **高精度マッチング**: エッジの形状を詳細に分析し、信頼度の高いペアのみを出力
- **自動グループ化**: マッチしたペアから自動的に連結グループを形成
- **日本語対応**: 接続情報を日本語で分かりやすく出力
- **Docker対応**: 環境構築が簡単で、どこでも同じ結果を得られる

## 技術スタック

### コンピュータビジョン
- **OpenCV 4.8.1**: 画像処理・輪郭抽出
- **scikit-image 0.21.0**: 高度な画像処理
- **NumPy 1.26.4**: 数値計算

### 可視化・出力
- **Matplotlib 3.7.1**: グラフ・画像出力
- **Pillow 10.0.0**: 画像操作

### 数学計算
- **SciPy 1.10.1**: 科学計算

### 環境・実行
- **Python 3.11**: プログラミング言語
- **Docker**: コンテナ化

## アルゴリズム

### 1. 前処理（preprocessing.py）
- グレースケール変換
- 適応的二値化による白ピース抽出
- 輪郭検出とノイズ除去

### 2. 特徴抽出（feature_extraction.py）
- 四角形近似による角点検出
- 4辺（上下左右）の分離
- マルチポイントサンプリングによる凸凹判定
- 各辺の形状記述子生成

### 3. マッチング（matching.py）
- 凸凹の対応関係チェック（凸↔凹）
- 形状類似度計算（正規化相関）
- 閾値ベースのフィルタリング
- グループ形成アルゴリズム

### 4. 可視化（visualization.py）
- マッチング結果の可視化
- 日本語での接続レポート生成

## 使い方

### 必要な環境
- Docker
- または Python 3.11+ with pip

### Docker を使用する場合（推奨）

1. **プロジェクトをクローン**
   ```bash
   git clone <repository-url>
   cd jigsaw-puzzlable
   ```

2. **ピース画像を配置**
   ```bash
   # pieces/ ディレクトリに画像を配置
   # piece_001.jpg, piece_002.jpg, ... の形式
   ```

3. **Dockerイメージをビルド**
   ```bash
   docker build -t jigsaw-puzzlable .
   ```

4. **実行**
   ```bash
   docker run -v $(pwd)/pieces:/app/pieces -v $(pwd)/results:/app/results jigsaw-puzzlable
   ```

### ローカル環境で実行する場合

1. **依存関係をインストール**
   ```bash
   pip install -r requirements.txt
   ```

2. **実行**
   ```bash
   python main.py
   ```

## 出力結果

### ファイル出力
- `results/summary.txt`: 処理結果の概要
- `results/matching_log.json`: 詳細なマッチング情報
- `results/connections.txt`: 日本語での接続レポート
- `threshold_experiment_log.txt`: 閾値実験の履歴

### 出力例
```
=== Group Details ===

Group 1: 123 pieces
ピース072の上辺(凸) ↔ ピース157の左辺(凹)
配置ヒント: ピース072を上に、ピース157を下に配置

Group 2: 2 pieces
ピース219の上辺(平坦) ↔ ピース232の上辺(平坦)
```

## 実行結果

最新の実行結果：
- **処理ピース数**: 249個
- **発見マッチ数**: 1,755組
- **形成グループ数**: 2グループ
- **処理時間**: 約109秒
- **最大グループサイズ**: 123ピース

## プロジェクト構造

```
jigsaw-puzzlable/
├── Dockerfile              # Docker設定
├── requirements.txt        # Python依存関係
├── main.py                # メイン実行スクリプト
├── modules/               # 処理モジュール
│   ├── preprocessing.py   # 前処理
│   ├── feature_extraction.py # 特徴抽出
│   ├── matching.py        # マッチング
│   └── visualization.py   # 可視化
├── pieces/                # 入力画像ディレクトリ
│   ├── piece_001.jpg
│   └── ...
├── results/               # 出力結果ディレクトリ
└── docs/                  # ドキュメント
    └── spec.md           # 詳細仕様書
```
