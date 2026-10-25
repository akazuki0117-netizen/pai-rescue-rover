# pai-rescue-rover

災害現場を想定したシミュレーションで、移動ロボットがその場で旋回しながらカメラ画像から人物（要救助者）を探し、
見つけた人物のワールド座標を推定して「報告」するまでを、安全確認つきの状態遷移として実装したプロジェクトです。

- シミュレータ：ROS 2 Humble + Gazebo Classic 11（Docker コンテナ上、headless 起動）
- ロボット：TurtleBot3 Happy Mini（差動二輪、RGB カメラ）
- AI：YOLOv8n（Ultralytics、学習済みモデルによる人物検出の推論のみ）
- 出力：ラウンドごとの観測 CSV、判定レポート（JSON）、状態遷移ログ

## 何をするか

1. その場でカメラ画像を数フレーム取得し、YOLOv8n で人物を検出する
2. 検出枠の下端（足元）とカメラの高さ・内部パラメータから距離を、枠の中心から方向を求め、`/odom` の位置・向きと合わせて人物のワールド座標を推定する
3. 複数の観測ラウンドの結果を位置でまとめ（クラスタ）、次の条件をすべて満たしたときだけ `REPORTED`（報告）とする
   - 同じクラスタに根拠となる観測が3件以上、かつ別々の観測ラウンド2つ以上で再観測
   - 座標ごとの標準偏差 0.15 m 以下、信頼度 0.5 以上、画像の端に接した枠ではない
4. 見つからなければ、承認を得てから一定角（約 0.92 rad）だけ旋回し、停止を確認してから次の向きで観測する

`REPORTED` は「シミュレーション上で定義した確認条件を満たした」という意味であり、現実の要救助者の発見を保証するものではありません。
`NO_PERSON` は「その観測で検出されなかった」という意味であり、その向きに人物がいないことは意味しません。

## 構成

```
scripts/
  person_geometry.py   検出枠 → 方向・距離・ワールド座標（純粋な関数）
  search_loop.py       その場での検出と座標推定を繰り返し CSV に記録（ROS 2 ノード、/cmd_vel は扱わない）
  spin_and_stop.py     その場旋回 → 停止 → 静止確認（--dry-run は送信部品を作らない事前確認のみ）
  odom_snapshot.py     /odom と /clock から位置・向きを1回だけ取得（読み取り専用）
  mission_logic.py     観測ラウンドの統合・クラスタ判定・次の行動の判定（純粋な関数）
  rescue_mission.py    既存 CSV からの判定を表示する計画モード（--plan）
  mission_runner.py    上記を子プロセスとして順に呼ぶ実行制御（状態遷移）
  test_*.py            単体テスト（ROS なしで実行できる）
sim/                   ワールド（人物アクター入り）、launch、カメラ位置を修正したロボットモデル
results/               実験の CSV と図（results/figures/README.md に図の説明）
```

### 状態遷移（`mission_runner.py`）

```
INIT → PRECHECK(--dry-run) → OBSERVE(search_loop) → HEADING(odom_snapshot) → EVALUATE
  DONE_REPORTED → END_REPORTED          RESEARCH → OBSERVE（同じ向きで再観測、最大3回）
  SPIN_NEXT → APPROVE(y/N) → SPIN(--spin) → POSTCHECK(--dry-run) → OBSERVE
  DONE_NOT_FOUND / DONE_SPIN_LIMIT → END_SEARCHED   UNDETERMINED → END_UNDETERMINED   ABORT → END_ABORT
```

## 安全のための設計

- 旋回は `--allow-spin` を指定したときだけ有効。さらに旋回のたびに端末で `y` の承認が必要（No・空入力・60秒の時間切れは旋回しない）
- ロボットを動かすのは `spin_and_stop.py` だけ。開始前に `/cmd_vel` の送信側が0、受信側が想定どおり1つ、静止していることを確認し、旋回中は回りすぎ・位置ずれ・データの途絶などで中断、最後に速度0を送り続けて静止を確認する
- 旋回の上限は7回。旋回後は `--dry-run` で静止を再確認してから観測する
- 子プロセスを起動する直前に、スクリプトの SHA-256 照合と、同じスクリプトのプロセスが残っていないことの確認を行う
- Ctrl+C では子プロセスを強制終了せず、子の後始末（停止処理）の完了を待ってから中断する
- 出力は毎回新しいフォルダに作り、既存ファイルを上書きしない

## 実行方法（Docker コンテナ内）

前提：講座の Docker イメージ（ROS 2 Humble、TurtleBot3 Happy Mini、Ultralytics 入り）。
スクリプトは `/home/ubuntu/`、モデルは `/home/ubuntu/models/yolov8n.pt` に置く想定です。

```bash
# 1. シミュレーションの起動（headless）
ros2 launch /home/ubuntu/models/house_with_person_v4_headless.launch.py

# 2. 別の端末で（端末を割り当てて実行する：docker exec -it ...）
source /opt/ros/humble/setup.bash
export YOLO_CONFIG_DIR=/home/ubuntu/models/ultralytics_cfg YOLO_OFFLINE=true YOLO_AUTOINSTALL=false PYTHONDONTWRITEBYTECODE=1
cd /home/ubuntu
python3 -B mission_runner.py --run                 # 旋回なし（見えなければ旋回せずに終了）
python3 -B mission_runner.py --run --allow-spin    # 旋回あり（旋回のたびに y の承認）
```

単体テスト（ROS 不要）：

```bash
cd scripts
python3 -B test_mission_logic.py && python3 -B test_rescue_mission.py && python3 -B test_mission_runner.py
python3 -B test_odom_snapshot.py && python3 -B test_mission_runner_heading.py
python3 -B test_mission_runner_safety.py && python3 -B test_mission_runner_allowspin.py && python3 -B test_mission_runner_approver.py
```

## 結果（シミュレーション、人物1人・配置1つ・距離 約2.9 m）

- デモ（`--allow-spin`、動画の実行）：人物が見えない向き（156.3°）から、承認つきの旋回を4回行い、8.3° で検出・同じ向きで再観測して `END_REPORTED`。報告位置の平均 (0.871, −0.886)、真値との差 0.129 m（図 `results/figures/f4_mission.png`、記録 `results/step19/mission_20261025_081118/`）。同じ条件の別の実行（`mission_20261025_074434`）でも報告に到達（差 0.145 m）
- 旋回なしの統合テスト：2ラウンドの観測で同じクラスタに根拠6件が集まり `END_REPORTED`（終了コード0）。推定位置の平均 (0.856, −0.839)、真値 (1.0, −0.9) との差 約 0.16 m
- 一周探索（STEP 17）：人物が画角に入る向きでだけ検出し、画角外の向きでは座標を出さなかった
- 推定位置は真値よりロボット側（手前）に約 0.1〜0.16 m 寄る系統的な偏りがある（原因は未検証で、補正はしていない）

図：`results/figures/`（向きと探索結果、推定位置と真値、誤差の内訳）

## 限界

- 人物1人・配置1つ・その場旋回のみ。検出できる範囲の境界、遠距離、遮蔽、複数人は未検証
- 距離推定は床が平坦でカメラが水平という前提。手前に寄る偏りの原因は未検証
- ミッション実行中に `/cmd_vel` を継続して監視する仕組みはない（開始時と旋回前後の確認のみ）

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) です。
人物検出に使っている [Ultralytics](https://github.com/ultralytics/ultralytics)（YOLOv8）は AGPL-3.0 で配布されており、このリポジトリには同梱していません（モデルの重み `*.pt` も含みません）。
利用する場合は Ultralytics のライセンスに従ってください。
