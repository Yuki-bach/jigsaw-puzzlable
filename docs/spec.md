# jigsaw-puzzlable 仕様書 v3.0 (最終版)

## 1. 概要
### 1.1 アプリケーション名
jigsaw-puzzlable

### 1.2 目的
1000ピースのジグソーパズルのうち、残された約200個の真っ白なピース（内部ピースのみ）から、確実に繋がる10-20個のグループを複数発見し、パズル完成を支援する。

### 1.3 開発方針
- **MVP優先**: まず動くプロトタイプを作成し、反復的に改善
- **高信頼度優先**: 確実性の高い組み合わせのみを出力

## 2. 入出力仕様
### 2.1 入力
- **ディレクトリ**: `pieces/`
- **ファイル形式**: `piece_001.jpg`, `piece_002.jpg`, ... `piece_200.jpg`
- **画像特性**:
  - 白色の無地ピース
  - 茶色の木目調背景
  - 統一された撮影条件

### 2.2 出力
- **メイン出力**: `results/groups.png` - 発見されたグループを表示する静的画像
- **補助出力**:
  - `results/matching_log.json` - マッチング詳細情報
  - `results/summary.txt` - 処理結果サマリー

## 3. MVP実装計画（フェーズ1）

### 3.1 最小限の機能セット
```python
# main.py の基本構造
def main():
    # 1. 画像読み込み
    pieces = load_pieces("pieces/")

    # 2. 前処理
    processed_pieces = preprocess(pieces)

    # 3. 特徴抽出
    features = extract_features(processed_pieces)

    # 4. ペアマッチング
    pairs = find_matching_pairs(features, threshold=0.8)

    # 5. グループ形成
    groups = form_groups(pairs)

    # 6. 結果出力
    save_results(groups, "results/")
```

### 3.2 ディレクトリ構造
```
jigsaw-puzzlable/
├── Dockerfile
├── requirements.txt
├── main.py
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── matching.py
│   └── visualization.py
├── pieces/
│   ├── piece_001.jpg
│   ├── piece_002.jpg
│   └── ...
└── results/
    ├── groups.png
    ├── matching_log.json
    └── summary.txt
```

### 3.3 MVP機能詳細

#### Step 1: 前処理 (preprocessing.py)
```python
def preprocess_piece(image):
    # 1. グレースケール変換
    # 2. 二値化（白いピースを抽出）
    # 3. 輪郭検出
    # 4. 最大輪郭を抽出（ピース本体）
    return contour, binary_image
```

#### Step 2: 特徴抽出 (feature_extraction.py)
```python
def extract_edges(contour):
    # 1. 4つの角を検出
    # 2. 各辺を分離
    # 3. 各辺の凹凸を判定（凸/凹）
    return {
        'top': {'type': 'convex', 'points': [...], 'descriptor': ...},
        'right': {'type': 'concave', 'points': [...], 'descriptor': ...},
        'bottom': {...},
        'left': {...}
    }
```

#### Step 3: マッチング (matching.py)
```python
def match_edges(edge1, edge2):
    # 1. 凸と凹の対応チェック
    # 2. 形状の類似度計算（簡易版）
    # 3. スコア返却（0.0〜1.0）
    return similarity_score
```

#### Step 4: 可視化 (visualization.py)
```python
def visualize_groups(groups, output_path):
    # 1. キャンバス作成
    # 2. 各グループを配置
    # 3. グループ番号を表示
    # 4. 画像保存
```

## 4. 実装優先順位

### Phase 1: MVP (1-2日)
- [ ] Docker環境構築
- [ ] 基本的な輪郭抽出
- [ ] 単純な辺のマッチング（凹凸の対応のみ）
- [ ] 2ピースペアの検出
- [ ] 結果の可視化

### Phase 2: 精度向上 (3-4日)
- [ ] より詳細な形状比較
- [ ] 3ピース以上のグループ形成
- [ ] 閾値の調整機能

### Phase 3: 最適化 (必要に応じて)
- [ ] 処理速度の改善
- [ ] メモリ使用量の最適化
- [ ] より大きなグループの形成

## 5. 技術仕様

### 5.1 主要ライブラリ
```python
opencv-python==4.8.1
numpy==1.24.3
scikit-image==0.21.0
matplotlib==3.7.1
Pillow==10.0.0
```

### 5.2 Dockerfile
```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### 5.3 実行方法
```bash
# ビルド
docker build -t jigsaw-puzzlable .

# 実行
docker run -v $(pwd)/pieces:/app/pieces -v $(pwd)/results:/app/results jigsaw-puzzlable
```

## 6. 成功基準（MVP）
- 200ピースから最低5組の正しい2ピースペアを発見
- 処理時間: 5分以内
- 信頼度80%以上のペアのみを出力

## 7. 既知の課題と対策
- **光の反射**: 前処理で適応的二値化を使用
- **輪郭の不正確さ**: モルフォロジー処理で補正
- **計算量**: 初期は全ペア比較でも約20,000組なので実行可能

## 8. 次のステップ
MVP完成後、実際のピース画像での動作確認を行い、以下の改善を検討：
- 機械学習モデルの導入
- より高度な形状記述子
- インタラクティブな確認機能
