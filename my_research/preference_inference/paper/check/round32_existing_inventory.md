# 第32弾 既存のものの一覧（計算の前）

- 作成：2026-11-10 14:55（JST）。読むだけ。`analyze/paper_extra/round32_inventory.py before`。
- 対象：第31弾と同じもの（`round31_existing_inventory.md`。復号器 `v12_fig5_s0`・格子のデータ・元のコードの Fig.5e の出力を含む）に、第31弾の結果（`Round31/`）を足した。第30弾の価値の網（`Round30/s1/`）は第30弾の結果の中にある。
- 場所の起点：`my_research/preference_inference/data/`（`.worktrees/v11` の作業木）。
- ファイルごとの sha256・大きさ・場所の全部：`paper/check/round32_existing_inventory_sha256.txt`（14445 行）。
- 下の表の「一覧の sha256」は、その場所の行（上のファイルの行をそのまま、場所の順）をつないだものの sha256。
- かかった時間：120 秒。

## 段階Iの予測器（L1 期間）

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/v12_s1_random_l1_s0` | 32 | 64,117,532 | `5481872720596b0d434c085082efc9c7ef5c6f5243031a11f1547e2c55e35c6a` |
| `result/v12_s1_random_l1_s1` | 32 | 64,117,772 | `2623a52029abf36ca379fbb9a17f2c7257e1f3cec63e1a815f06abbfacca30c8` |
| `result/v12_s1_random_l1_s2` | 32 | 64,117,532 | `41f5457b2a8dc8b54ea2921466d75e6bead054f05ebfe7f21e4917c2b9838c57` |
| `result/v12_s1_random_l1_s3` | 32 | 64,117,532 | `9d8f266fe6993ac7dd0c2e713c3686ec1fe447708591f004e49a141d57f4e21a` |
| `result/v12_s1_random_l1_s4` | 32 | 64,117,532 | `da510b709fb98f51e43c39bd01921a6918de05dbcdcad9fa8d01a62b0e4af16b` |
| `result/v12_s1_random_l1_s5` | 32 | 64,117,532 | `ac94d75c8fb93034a442fa8f7379f8a2739213a3d684dab770860d6f7ec17110` |
| `result/v12_s1_mixed_l1_s0` | 32 | 64,117,384 | `b17b4efae44c6b970993674c982792422101cfa945cbc701bddb21d6953daa4b` |
| `result/v12_s1_mixed_l1_s1` | 32 | 64,117,350 | `ca90521f200c7727234741c99b1b94ac0d2341eccf79128a3abde01bc5c9e799` |
| `result/v12_s1_mixed_l1_s2` | 32 | 64,117,385 | `25848ef64fe93a659b221a3b6f558119fb30f2353c727cb77bfab1c82e02d598` |
| `result/v12_s1_mixed_l1_s3` | 32 | 64,117,372 | `26a8c0cb39462b91ff09e380daaadf511a2ef556e9d894dcb2bf2b29f0f69c03` |
| `result/v12_s1_mixed_l1_s4` | 32 | 64,117,384 | `459e0944ea92ce4a3417a1f6565ac09879f8eae1228e50597f5c03ba5a0d6df8` |
| `result/v12_s1_mixed_l1_s5` | 32 | 64,117,349 | `b2708319d7187e2db36fb72d366cfd9d1c36db42dfbc7b41f1fc30506271041f` |
| `result/v13_s1_mixed_l1_s0` | 32 | 64,117,292 | `2430bff4379acc5349ae2ee49a2205b63c0fc39b62f23d68a487998952880e83` |
| `result/v13_s1_mixed_l1_s1` | 32 | 64,117,292 | `0a547849604ab1bfb4df695ef1468eaaa8770f5b123598cfa6394d4a752ae73c` |
| `result/v13_s1_mixed_l1_s2` | 32 | 64,117,292 | `0c1a02e84a370bbddb2cde41da5e371e058fa859d6c5b10f28d9b898a0d0c790` |
| `result/v13_s1_mixed_l1_s3` | 32 | 64,117,312 | `3f9877bb9c6dbe0e4f69a574da5d9b88cbc58df8a49311e28d0830aa339c02ec` |
| `result/v13_s1_mixed_l1_s4` | 32 | 64,117,292 | `37e73242f33a66397dfe72e2e22f6c9d476450d154e16a7a4aff1766bf6f31d4` |
| `result/v13_s1_mixed_l1_s5` | 32 | 64,117,292 | `bf042f9aa276e13b91e1cc3f6cddfb2b27d4c9f0c49af7802fa8f8e90b85be24` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s0` | 32 | 64,117,484 | `93e9ab69d82c2bfd10f0eb74404d09c3e2e537081b88cd54eecb76c8921b4527` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s1` | 32 | 64,117,425 | `bebc0d0d552ce6a8a3fce2e89af6653d111f3f4f3f3cdb0d24ed235692b92438` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s2` | 32 | 64,117,475 | `1e3065d0a096fc57b5a205462548e79bc1dc361c66c600711918a35a2645de07` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s3` | 32 | 64,117,479 | `356de9c5d2b0fc56542630452f03421b5eee3febfa46869929b79bfb00a7c6be` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s4` | 32 | 64,117,474 | `bbc5d43d0acd64c4e7b2ae574e845e26367016019766c9cd8a884e89bade054c` |
| `result/paper_extra/E1_signrand50/s1_signrand50_l1_s5` | 32 | 64,117,530 | `d5570723478464756e08409274c172338471bb55ddc3841742404a67524a9d90` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s0` | 32 | 64,117,373 | `19e29c500fc7a95f80b5ca3f774461d6d2421992c6dff62dd6954d124e2957b0` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s1` | 32 | 64,117,373 | `e9775bed033f389bdc5ad34fa900546ee06937a0ee18d0c6bfffc38723781811` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s2` | 32 | 64,117,373 | `0d6fa36b52144e7b158b9d888154d7d951da91a8c198f9ea6791c6fef4277a59` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s3` | 32 | 64,117,373 | `cce4e77eebe2bd3e6379ae811de9dea0e6c2bc7395d16f5054f75ef9dbcb3f79` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s4` | 32 | 64,117,373 | `64e6c7b3218b43314da33dd5a84198ca743fb3cf495ea6d43ed00acb3c3b44c4` |
| `result/paper_extra/C_signrand/s1_signrand_l1_s5` | 32 | 64,117,373 | `2b573aedd356a71708633dfb164f54d32b9eff3bad7d431ba4a69e769b01bf15` |

## 段階Iの予測器（MSE 期間、L1 期間の出発点）

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/v12_s1_random_mse` | 1266 | 4,902,577,224 | `d758ea54083e60cddaf83bdc50397024efea5462d57ab75968276642c4c9089d` |
| `result/v12_s1_mixed_mse` | 1266 | 4,902,576,738 | `bbebb709c1a28342ab6279438e7ece5c2eef594261a237dbdb36a225b4e73d92` |
| `result/v13_s1_mixed_mse` | 1266 | 4,902,576,698 | `45556037958dd3d2571f8efb3f1b80faf65dad2be788e9a20e604d91052025a5` |
| `result/paper_extra/E1_signrand50/s1_signrand50_mse` | 1266 | 4,902,577,227 | `30f264867a3fe06d41662cf356fa91457ae847c5a520cb6d190c8ac28f0417b9` |
| `result/paper_extra/C_signrand/s1_signrand_mse` | 1266 | 4,902,577,144 | `6c0666b0f039b203352b7a1a447371821c0f0176365d45dfae648fc599e1f7c4` |

## 段階IIの生成器

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/v12_s2_pref_mixed_s0` | 32 | 68,545,756 | `d6ca50a73227369a72be83b87628f68d6bd0857f67d00b36447fc886349d2b41` |
| `result/v12_s2_pref_mixed_s1` | 32 | 68,545,756 | `37aed5f18add5437b2ab37965a3026f9714f87ee829d800a41f27c3a31a584c8` |
| `result/v12_s2_pref_mixed_s2` | 32 | 68,545,756 | `85eb89088f3e3d2da5ad7167985fded6993b3b329232096051f77467477538fd` |
| `result/v12_s2_pref_mixed_s3` | 32 | 68,545,756 | `0d1ba4328610f833654ad8d4f5144b4497cd62a22eed7948f892a8dffbf16c5e` |
| `result/v12_s2_pref_random_s0` | 32 | 68,545,756 | `f4f6913825c6d2fea2e5f1f3daf98540707fc31aa01c26e49b70e093a571866c` |
| `result/v12_s2_pref_random_s1` | 32 | 68,545,756 | `7acb1fbc3699b6ca60d768e4152f82f48089fb06935a6406aaa654ad9e695de4` |
| `result/v12_s2_pref_random_s2` | 32 | 68,545,756 | `6fb454166db3a945b21b763fd845195061d37ca64fa3576234ed2b4a32becc80` |
| `result/v12_s2_pref_random_s3` | 32 | 68,545,756 | `445ccd8ebae942b7195efedfa907964b875302701609a2301adaa9c46debc307` |
| `result/v13_s2_pref_mixed_s0` | 32 | 68,545,756 | `8b816e2ce9900ed2c415be096ba0baeda4a0ea2c1889a04332f863e2b5428e81` |
| `result/v13_s2_pref_mixed_s1` | 32 | 68,545,756 | `606b2a8acf04b0a4bfe4a7c81469243853b1a09e97d7ca92163694f39df4c4bf` |
| `result/v13_s2_pref_mixed_s2` | 32 | 68,545,756 | `1c37da5494242c5f1239307bcfb82c6c7d663e6f01625d4154a5f5a157068dc7` |
| `result/v13_s2_pref_mixed_s3` | 32 | 68,545,756 | `7666211776999f67985aa25f28de378684c0f2552974ad696021c78efddffa70` |

## 第20〜24弾の評価データ

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `data/paper_r20_swap_p80` | 4 | 85,944,158 | `8f75678a93bb4a2cbdba6b9e3f175c2a2ec9ec99987aa4f0bebb7ac9b9e60bf8` |
| `data/paper_r20_swap_p50` | 4 | 84,450,739 | `16f19fb8d1011d334981de67e53bd581e1e78c9df1ffe3110654f4818c894179` |
| `data/paper_r21_swap` | 2 | 17,685,450 | `7bf3173562b96cd73cae131d7d813a87ad55552121146cdd4274435c96bf8b18` |

## 第20〜24弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round20` | 284 | 2,334,361,497 | `64fb4753768ba8042c5daa3e70803e47a684262e3b11064d798d9a97ecc89cbc` |
| `result/paper_extra/Round21` | 296 | 212,476,805 | `8cba808b1d5d0daff9d8ac02229c71c3575b47923d8a848015b58b8714b68c87` |
| `result/paper_extra/Round22` | 408 | 304,970,851 | `27662c8b70d6515614276f112bde146034e5d1b6031763a695edb4373f7ea178` |
| `result/paper_extra/Round23` | 598 | 953,998,527 | `88341815195112229b44c6559c8dd51ccc4c35bcad98a5847d49963f6aaa012d` |
| `result/paper_extra/Round24` | 71 | 22,188,397 | `3788abd538f416f3f9462fe575aad58f8bf241d256fa701ed87290759a59ec27` |

## 第25弾のデータと結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `data/paper_r25_swap50` | 1 | 2,059,490,284 | `d53b99adb15aa4e414789c3afe15cb397f7e6311b8f271afb5f79f55daa60cc8` |
| `result/paper_extra/Round25` | 673 | 596,355,422 | `872979bd0753df04ccb6d1f57cd5f71c6ca28ecd9114a46fa88e4f31e9971133` |

## 第26弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round26` | 550 | 63,142,414 | `3a69833d620f0e4c6eb089b3f06b9873a677d57275a58ede1d57c8978b8df60a` |

## 第27弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round27` | 285 | 58,730,657 | `3894e4214c3bfc2162c6410090e07b75f87f9dd6ec87b485cff424d0b024656d` |

## 第28弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round28` | 637 | 78,829,354 | `85d76ed602e274f58210ea190f24546f196cbe8cb4ba95940a5c325dee60fb3a` |

## 第29弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round29` | 197 | 119,150,820 | `9f61ed62eea406c875aca1053b436e29d8ba4b421a3f2feebb2604b2a3143c47` |

## 第30弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round30` | 78 | 45,381,247 | `5abad8718b991bc7b5ee139b4a2e6d7ee5dc684db492e4ad22336ed0fbaf58b8` |

## 第31弾で読むもの（復号器・格子のデータ・元論文の Fig.5e の出力）

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/v12_fig5_s0` | 104 | 229,263,675 | `264e880082846b24e1d65aa18597e4a7f96a906cd510df8605d7561ebcff2874` |
| `../../../../../data/data/grid` | 1 | 9,562,245,776 | `8aaf7b51dcb0399ecaf3fe6dc34b5a0626a12dce84f017af6467e8b5ff996ed0` |
| `../../../../../my_research/preference_inference/data/data/v10_swap_grid` | 1 | 2,392,902,504 | `7e112f0e89f9dff47b077e15705afc2ad442a5f968f226d2767e73aa6f920e31` |
| `../../../../../data/result/exp2/0/test/grid/save/vpt` | 2564 | 237,623,451 | `0edeaad0d43a178d0930167aba507e1546d781f7be99da2c63ff4e3ce0c506e2` |

## 第31弾の結果

| 場所 | ファイル数 | 大きさ（バイト） | 一覧の sha256 |
|---|---|---|---|
| `result/paper_extra/Round31` | 13 | 303,183,769 | `c170869d8919d70dcbd302b77ad4eb55b75861d5eba738a05eab5320bf08e4e0` |

- 合計：14445 ファイル、47,021,332,433 バイト。

## `/mnt/md0` に移した古いデータ（場所と一覧の確かめだけ）

- 場所：`/mnt/md0/kusano/superposition_archive`。一覧：`/mnt/md0/kusano/superposition_archive/archive_md0_2026-11-02.sha256`（7914 行）。
- 今あるファイル：7914。一覧にあって無いもの：0。一覧に無くてあるもの：0。記号リンク 10 個。
