# Original patch and native/build file catalog

Full SHA256s below identify original bytes. Dispositions and application order are in [CAMPAIGN.md](CAMPAIGN.md). A hash match is not a new build.

## Campaign patches and diagnostic diffs

| Original path | SHA256 |
| --- | --- |
| `evidence/hc-v032-v104.diff` | `40a15e9ab72567c3a6e15dd29c5d1137d1e266d58f99c30d83f7562296c7268a` |
| `evidence/phase4/incomplete-patch-detected.patch` | `0ad9a6e85e3cd8802e7760cd9f25ceedabee627d2c8d392a866d7b6a115d86fc` |
| `patches/astra500-beyond.patch` | `540ef96c87e99fc4617b2b5571526cb31a403da8560ef0cf15b8b8159bfd3ac8` |
| `patches/astra500-consumers.patch` | `4a73422edfc2f8c304966b365832b65e85e7436b49e82d87da375de44b049738` |
| `patches/astra500-cycle.patch` | `1650615ea69f3f0eeb3c72cf25566f221383e08ebccf487a6bbf8f39bccee778` |
| `patches/astra500-dominant.patch` | `8a97ceed1adeb1b4df148e8c2b4f8c735529b6351b3dd03e99900410fd26b0db` |
| `patches/astra500-ownership.patch` | `36887d6cb787f125ccdcb97cce30b85ff180b598973d1ad9864d9d431d8e4919` |
| `patches/astra500-structural.patch` | `b57de90e36b8b1867142c7693885e60413ab83d9993a5fda3fd2fb0cd9379725` |
| `patches/astra500-v032.patch` | `ce19b3bc0531f86507e734ece0530616a3405b98efeaf2a98ffaa157af33c13e` |
| `patches/deep-c1-v032.patch` | `d1dae47ea1db3e47c3eee1fa68388a5b1458ea8c724955a91c502e66bdd65f6b` |
| `patches/dflash2-v032.patch` | `cc9bc66d5645ab293b0c9310c66820f6923e22e0b3998ab7fe2cdd96fa0ae03b` |
| `patches/fullwidth-c1-v032.patch` | `412d7cd022a5008e86e23b568ee2df1edf57ccf94f5b64889fd4e58985f50bfb` |
| `patches/integrated-serving.patch` | `6cd8586e496f0dcab0513b14742a758dfc1d1348cd7f339bb2f8be42a2d19be7` |
| `patches/long-prefill.patch` | `391975bcd98a90a000038283ecb96b98c3cdf26e233cfea97354acfb38b2b4e1` |
| `patches/mtp7-gap-v032.patch` | `9106f52abba30649fed4f3f05076de8501147a3fead33a606c02c0f718aa9425` |
| `patches/perf-c1-v032.patch` | `8565a95f99b221a923c6ed4551609a355b58e417ee9ad6d1ac8571c1f3e934ed` |
| `patches/prefill-serving.patch` | `efd7de9122ca30bd9e1723d10b083a156e21a839da6837a544b9a018329878a9` |
| `patches/target-v032.patch` | `b6907e3ab1921cb136852f9af7a6e65605e200a6ed488597fd7f6e25c5e8d6f6` |
| `patches/verify-c1-v032.patch` | `6ab5ddaf5e109b22839622ecbdea463604a83f5039b6b88f86b0c055c7222566` |
| `external-draft/patches/qwen38-dflash2-capture.patch` | `91708e18c67f1c0e8f6c19cfbe7cc9f59ff257c05d61bb30b407ae4c36354169` |

## Every custom native/build file

| Original path | SHA256 |
| --- | --- |
| `build/astra500-beyond-gdn/beyond_gdn.cpp` | `9b5d56f6ff617608f596e0e5104a44ef4dd6a91c9d049cf39c7ecb9cd7aa939f` |
| `build/astra500-beyond-gdn/build.sh` | `050ecb051f01b3bb9c282e65d6ea1f7ec1f49d679d9246a384e3e28fa8784503` |
| `build/astra500-beyond-gdn/project.hpp` | `750697363340543eef974bcbe1cf1208c87c5d5953dbe4a186b701d42ddbfa40` |
| `build/astra500-beyond-prefill/beyond_prefill.cpp` | `465be5170d77338c4dec09508a319f0176cebbad34cd3f85a0b84f850d9b9477` |
| `build/astra500-beyond-prefill/beyond_prefill_v1.cpp` | `6eba67753040e3992a949854a46932118b15bb25d3f9a7d9c9debf26d1556828` |
| `build/astra500-beyond-prefill/build.sh` | `c4809a5c927ca533eda3d5fd7d1e53e7b561165f8cb5443dd39eff637310c5ed` |
| `build/astra500-beyond-prefill/direct_gemm.hpp` | `fcbffdc8086c21899770497a9af85b8b8defa1a922c1fddcc92a804899339d94` |
| `build/astra500-beyond-prefill/direct_gemm_v2.hpp` | `a723129d94d02a9aa0b7c36d0ac87d750d3622c6607c4346f823a4bccd37a688` |
| `build/astra500-beyond-prefill/generate.py` | `6413f1bd9dbdee512429f27014e998706f289458c5686c9ae49c67fe33d063b3` |
| `build/astra500-consumers-native/build.sh` | `48f8b7b388b7c307fd58980260cf7bb98b2ec878f1556ddab8eea84adb96ed0d` |
| `build/astra500-consumers-native/consumer_gemm.hpp` | `6c48c9935e851ce9541bcc3c059a9d82a0da6b4edc59231d50b4309c8066cf48` |
| `build/astra500-consumers-native/consumer_gemm_v1.hpp` | `7000a13fbe9f5da616e2268e89d2fed3ac442ddc89520dc4fefb6135dc24c0f4` |
| `build/astra500-consumers-native/consumer_gemm_v2.hpp` | `1e91d3cc028ac3b5cdc3b2f8b4bd45cff9c91d4b408dc226af1e16b04fd64bd5` |
| `build/astra500-consumers-native/consumers.cpp` | `05677bfdf892ff632a0ee1d1854a317b3fc167315593b035a57dc24b7d75e720` |
| `build/astra500-consumers-native/consumers_v1.cpp` | `1cd71ece3e9596c6da8605d2b9e8947d1cc1d9e145623930fd07d39aedfc99b9` |
| `build/astra500-consumers-native/consumers_v2.cpp` | `0927bba581bb48bf5164820f54ca052e2c15047b5a1787b16bc5bd2427f2c4c5` |
| `build/astra500-cycle-gdn/astra_gdn.cpp` | `2df8bfb60a05f3a6f69c6546be54453c4f0ca3436b2c260a0209938ee0cb98e7` |
| `build/astra500-cycle-gdn/build.sh` | `f49548041c7beb46f63cfefbdd76a021124cb02ab61a35ec4d2dd8e7ff855b18` |
| `build/astra500-cycle-gdn/generate.py` | `4609687c225debe50dc6a1e40273e9b13197218f2424bb02d4652ba1acaa7b1f` |
| `build/astra500-cycle-gdn/prepared_conv.hpp` | `d75a7273fe9032359197398fd52fed1b88373f8299ae5c8a14aa8ef85b548e2c` |
| `build/astra500-cycle-gdn/prepared_recurrent.hpp` | `cb3cab7fadc24c5cdfefb261440cf583a193c0294c649467b44549c22fdcc86c` |
| `build/astra500-cycle-hc/astra_down.cpp` | `e7105a4a20aa3f2a4d74af9cb2edacbea54a2508c95ccb000e0e1461aeceaf15` |
| `build/astra500-cycle-hc/astra_down_v2.cpp` | `ba90b03f704a1f395460050e7bbd5b5d3b8e6bad8b3e72073848175199af0bf0` |
| `build/astra500-cycle-hc/build.sh` | `8e5932c8bbb0f62ba61625f32c373f0704c2f4c24ee1146293e686dee4517d4a` |
| `build/astra500-cycle-hc/down_tile.hpp` | `7bbf9ab9e2bbb41dd2bcc815a3741e82ce70f606f4fefae8e7c0737713a9cef8` |
| `build/astra500-cycle-hc/down_tile_v2.hpp` | `920bffe31f821cd254b991e825cf1ee2275728ba4427240779e9d2a69bc68ef2` |
| `build/astra500-dominant-native/build-dual.sh` | `c97d0c106d2609a81be9a8c71538936485e1aca73e7715753dc373b507a6d469` |
| `build/astra500-dominant-native/build-profile.sh` | `23e90179ef415ff50e8435d12ec6896e3af958d4c221a55abdfc5eaaa3e3df49` |
| `build/astra500-dominant-native/build.sh` | `fa59fb0a25137a424011e166518415921e0122516991deb6215188ede1f590d0` |
| `build/astra500-dominant-native/direct_fast.hpp` | `ac6bdf75a2ca9f1eaa39fbe85c3654bd2b39bb235090b0620285709dae209390` |
| `build/astra500-dominant-native/direct_gemm.hpp` | `67b64f22290b9d1c821ca586aeb473973196f5525acbc208aca9a86ee26291b6` |
| `build/astra500-dominant-native/direct_profile.hpp` | `67b64f22290b9d1c821ca586aeb473973196f5525acbc208aca9a86ee26291b6` |
| `build/astra500-dominant-native/dual_gemm.hpp` | `d6698e8f82e08bf31300470a7a41eb2af0ad7f6738b2bbca8f0db654bd01e6d2` |
| `build/astra500-dominant-native/layout_gemm.hpp` | `7d6ffb1eec6992f99238f5e8c380707605a961656c1d4134a4080e14aebc4a9b` |
| `build/astra500-dominant-native/w4.cpp` | `dbf25b4f5ad3ff26469a6d33edd90e13b751fdf767fb4f0fac22bcd6857e23de` |
| `build/astra500-dominant-native/w4_dual.cpp` | `dbf25b4f5ad3ff26469a6d33edd90e13b751fdf767fb4f0fac22bcd6857e23de` |
| `build/astra500-dominant-native/w4_layout.cpp` | `4bef7018a71ee1bcabb18f88e106a5730b4ea42253dc6cae13251a1da9003998` |
| `build/astra500-dominant-native/w4_profile.cpp` | `11bd940efe0e312e501b274403644747de5ff38b7a5a9220ac70653ca9114b8a` |
| `build/astra500-native/astra_moe.cpp` | `38dd2f12da21a143a6cfa7050d630e460954cfa06499c672d7023b554cd08baf` |
| `build/astra500-native/build.sh` | `8d77878d421d06705a50169f13adc3bbf4430759a0454d498cd83d45278316f9` |
| `build/astra500-ownership-batch/astra_batch.cpp` | `edd368bf0203085ced760ec66d47071986a0d120c2a2e01128ff85f56ff0b8fb` |
| `build/astra500-ownership-batch/build.sh` | `f50734ad6f6ce3a69e46513848c89657c22e4d1a9fbadb5190488bf1f12e0719` |
| `build/astra500-ownership-batch/direct_gemm.hpp` | `b29bf23c5e9d706d33b0e35bbd79fd922ade9b1b36eb5d5e42ae4f90dd78a167` |
| `build/astra500-ownership-clock/build.sh` | `6ac86d598ca9932dd5d63e5ee18330cdd433ddfe6cbdc82ca69e76ed8f26ee77` |
| `build/astra500-ownership-clock/clock.cpp` | `89e5f28fef9a42f3f63b2b5e64176398a01cb400983d6b856fa8e6dfe85bf3f1` |
| `build/astra500-ownership-clock/trace-legacy.cpp` | `e36e5e70f77be0f454d44f1ac5b5c07ac2183e83d1537852e125eb81e04bad34` |
| `build/astra500-ownership-clock/trace.cpp` | `9e5cf3aa334a5a2466ebfa1e002f82cc9216e131510a12ae0395d5063164d464` |
| `build/astra500-ownership-exec/astra_direct.cpp` | `db199e64a4e6520985076c58f89ddfdaec830ae9e65f73b2419bc8edf9abcfd5` |
| `build/astra500-ownership-exec/build.sh` | `c540f690d026778afe17652e11eddf1cfef25b4d999aaa583f813904fd316c86` |
| `build/astra500-ownership-exec/direct_gemm.hpp` | `717faeda5bf007e090ffbd80fc8e4354ff7e10d64b6f31e5b24a12a9b22d7bac` |
| `build/astra500-ownership-groups/astra_direct.cpp` | `b4dd8789ab0bd2b8f9fbc22809537f48c2f98a7feaaff0d3cbbc443176fbd637` |
| `build/astra500-ownership-groups/build.sh` | `dbdee0862d06fdc4d4bc6ddb9f5e9dd7e0b9ba6508b0fc7608e829d95369269d` |
| `build/astra500-ownership-groups/direct_gemm.hpp` | `512417e6b0ba4e054cff6f2626da28d126c21534c7bbef3d55777e5d4494f25f` |
| `build/astra500-ownership-groups/ownership_gather.cpp` | `035f9952644e9dbe77b17e5c6ad6b519cf4d2601d2e334daa4cf8fe0725b9b1c` |
| `build/astra500-ownership-groups/ownership_remap.cpp` | `ada545d108dbc5c709ba8ef44c931b41416b5a0011d950aadeeffeb0a6938c6f` |
| `build/astra500-structural-hc/astra_hc.cpp` | `446fd75c4216f08c3f2db87efa39f049c467f729797c44b54c778b3959fe5f42` |
| `build/astra500-structural-hc/astra_hc_v2.cpp` | `ca9601dcc29209049bc45405ee96debe8acdf113520ceac5ee36ff4297a43094` |
| `build/astra500-structural-hc/astra_hc_v3.cpp` | `6f03646101a9adb093b7846c394eb4af913496ed034357db0c5305d7ef215b5f` |
| `build/astra500-structural-hc/build-v2.sh` | `9ed572795750eb80b761030ee15e1f86f1421ed2cc19c0c66bf9e7525dad0717` |
| `build/astra500-structural-hc/build-v3.sh` | `4c9122c3eaad899f56d0fdc7a467781d5146c218a5d476fd63feccbc34122050` |
| `build/astra500-structural-hc/build.sh` | `c403332e5353ba854b02a77a878b6618e8d3a4c0d23d69b625cd85ad5ebee090` |
| `build/astra500-structural-hc/fused_hc.hpp` | `1e1c5cfd4f9316e09c3eb7fca2acfcfdeb3bf4cb5a1b37755876ee597ec79a4d` |
| `build/astra500-structural-hc/fused_hc_v2.hpp` | `454e5f9873c9a0838874505ca46a082610d7c196bcd679d88f4f09f497c0be63` |
| `build/astra500-structural-hc/fused_hc_v3.hpp` | `d207bf1399aaa3529fb71083561b18050a7250540ea091311210b0c44a761af9` |
| `build/astra500-structural-hc/generate.py` | `bec3ed149d5b6753028f01e3cc7c7394c899bab52ce832ebbd0d7a585488d084` |
| `build/astra500-structural-moe/astra_direct.cpp` | `3949c45afcb72ef94ea51076038d4b05fd3ff604c3fd2679dac11bd1b60823b4` |
| `build/astra500-structural-moe/astra_direct_v1.cpp` | `6294b6496448b176bb0067193bc30b8bc8985aaa87b03d5fc0721475aaa17546` |
| `build/astra500-structural-moe/astra_direct_v2.cpp` | `5de2d7b5e5c9bdbf00b6db1eb0e3ac0c04431e20bf08cf36a6d1d7cac75b7e88` |
| `build/astra500-structural-moe/astra_direct_v3.cpp` | `7cb93c5e4f62caa82f5713017f6994d962ff473baa48699c8c87e502b6fbf5b1` |
| `build/astra500-structural-moe/build.sh` | `c774c20c2baa71455831e7cd51f80f24e5eb84fdb1f926370dff1f2d26ac4d63` |
| `build/astra500-structural-moe/direct_gemm.hpp` | `37b58dd9f567ef06df016df1e5a3247742ad4e8385edc2ed53a79940921d882e` |
| `build/astra500-structural-moe/direct_gemm_v1.hpp` | `997693b54cdc829c773bc80333fbe3793f370e5fc5e3207b49d1ada166a135df` |
| `build/astra500-structural-moe/direct_gemm_v2.hpp` | `37b58dd9f567ef06df016df1e5a3247742ad4e8385edc2ed53a79940921d882e` |
| `build/astra500-structural-moe/direct_gemm_v3.hpp` | `bb44a73798be6ca22cc1d63cf365a5a6edb071c65c852db77bfcee83d44a7d62` |
| `build/astra500-structural-native/astra_ar4.cpp` | `c6e318583a49da6924d67ad682e9ea701c1cb925d896523c3aac3dce7dfe04ca` |
| `build/astra500-structural-native/astra_ar4_v2.cpp` | `1de3b1f53862d59d5c46f1958917e0b8a2238afadba6b7ec6e3af178ab828aea` |
| `build/astra500-structural-native/astra_ar4_v3.cpp` | `e388c7efa65c6771f9f3b896663c4dbac5d6a2710ca16d1cfa6e866efc42ace3` |
| `build/astra500-structural-native/astra_ar4_v4.cpp` | `60633bfd235fec880c56aca8d14811f8a5b6b44aa49c861517a5d141d172804a` |
| `build/astra500-structural-native/astra_ar4_v5.cpp` | `cb75f79101523969c9703c1e3ff86f31d92053160f8d7f6ba43fac4da282500f` |
| `build/astra500-structural-native/astra_ar4_v6.cpp` | `641fa61a4e21bd9121e3b7012802f9c9cd5ff271d42377f57db9648d49401aab` |
| `build/astra500-structural-native/astra_ar4_v7.cpp` | `9d076f5316529d2c9f428b005f413a434c77f5efa36ae39c24a9365737f1bc6c` |
| `build/astra500-structural-native/astra_ar4_v8.cpp` | `cc2848e79687b09a6bfc34eaf8df8ad3ba26c6c79653cf90c86f42a15c2bd3ea` |
| `build/astra500-structural-native/astra_ar4_v9.cpp` | `9881e2f63dfeecbf2ecfc82caebeffefed3f4d3bc55712ecee7bac9aa194329f` |
| `build/astra500-structural-native/build.sh` | `f83f28b31d4e15beabd47faca86b3abce10fcc903e817d7350c3db42ce1b209b` |
| `build/frontend/prepare-serve.sh` | `3815a1c2fd3135163ef8ef47b06bcf36808e4a1dda42419684273fc1458b5c34` |
| `build/integrated-native/build.sh` | `4c8b5334f99ebefe11417d1f1f6a08411fb7824a409ddb9cbe37df4e26acbcb6` |
| `build/integrated-native/direct_gemm.hpp` | `f43a0de2bc01fbaa5202624038b5412e16509243e88baa38db96c3b0e75c3b0a` |
| `build/integrated-native/hc.cpp` | `8b745db1f7cad919ee31004cad9f882905d10ec2a4359be58f001f051b3e8b64` |
| `build/integrated-native/hc_tile.hpp` | `c699a3870055f8bec51aff8487da38ed592c6d280e8f575a7feb0e2e5ced19d5` |
| `build/integrated-native/prefill_small.cpp` | `07b1c1d557b5b3874bc032a12990f720aca3d4e3cc8a6e6c2a2f871d4a0249d4` |
| `build/long-hc-native/build.sh` | `addf3ab3d7c54b5f3bf2bea855ebe318bca6efbdfa9cb6c572341aec6a1a4634` |
| `build/long-hc-native/hc.cpp` | `2670e87ffda93f19f3584f42a365b42e39270cf38352e244c96b9a9699ab71ca` |
| `build/long-hc-native/hc_tile.hpp` | `fc3e6ae6357924a0190acdd3500334b50fe221afbcb3b47b3a7e36c1096ea89f` |
| `build/long-hc-selected/build.sh` | `2805ddbb389f984df6c121dcc454f4c9268ec086773d0cb89acd4613fd83a7b9` |
| `build/long-hc-selected/hc.cpp` | `3fbfd691fd2efffe69fb3d36b96c648abaccf27c4e57095d49537af06b77f969` |
| `build/long-hc-selected/hc_tile.hpp` | `8c57f69658c0c7354f8bfd4241bd082373504af400bbccfd0b32a37351959f96` |
| `build/long-prefill-native/build.sh` | `1a40b7cf1d6dd3d0b4cecc459408b4fae54cce716328ccf1b665a0e4f45cb0aa` |
| `build/long-prefill-native/direct_gemm.hpp` | `a4f7f6c04cdabc9d7a910b4372063bdadd5e1dd1dfb61fc5bd56e5548d6a46d4` |
| `build/long-prefill-native/prefill_small.cpp` | `2166316cc51171535610f00201bb9d112eee4d5884af98b5b8cea1280d3dbb21` |
| `build/long-prefill-token/build.sh` | `c01fbd4e9d11db7b0f8332b2fb5778ee2646c4018a361b757ed3f641f7cf9a0e` |
| `build/long-prefill-token/direct_gemm.hpp` | `80172dd4a836c65d781b2987a01f3c65b41a9deabdee09f1acce834f69444c67` |
| `build/long-prefill-token/prefill_small.cpp` | `11a8d2f4266f2333a632f71697be32b05645f42d0ca04bfccc4b6617b24e0cc7` |
| `build/native/build.sh` | `419c6746028d5c4dff704ae2b2108c8563a52b595c20b7a8e5f8ca94084271e2` |
| `build/native/gdn_index64.cpp` | `8de6cea7fe553900c1c18d923a1f32b4ee1d42f3c6118cf93cac31e25b7add84` |
| `build/prefill-native/build-medium.sh` | `02b09c1eeb7292f816fea32a2b20fcbbf652c8bf13ad5d966e14293b500b7919` |
| `build/prefill-native/build.sh` | `73b6a5784308382c21bc630613b7cb5447eefe08b788297ac02e3c82c49eb961` |
| `build/prefill-native/direct_gemm_medium.hpp` | `0cb63506926d8123ce9bab4bb9da62fa102f112d610056cf5ce476486e42e6ff` |
| `build/prefill-native/direct_gemm_prefill.hpp` | `c426cd3704f3c352c18c922ca5c1eea1946dbc720869e29837d776716f61ba0a` |
| `build/prefill-native/prefill_medium.cpp` | `2e92d76be4b97f74e4005c2fb459495a547814acce7554358c45557d9502fe66` |
| `build/prefill-native/prefill_small.cpp` | `69ce2f7d226e106150d05894dc1c1d03ebacee503f6a65a225c5dfbe33605a82` |
| `/home/wulf/Desktop/Benchmarking/flashnext-long-prefill-20261021/hc-v1.cpp` | `3fbfd691fd2efffe69fb3d36b96c648abaccf27c4e57095d49537af06b77f969` |
| `/home/wulf/Desktop/Benchmarking/flashnext-long-prefill-20261021/native-v1.cpp` | `a11aad45c8cb05a3ef4a047c3aca74505daf6f771dbd014bd7b428ddd79df003` |
| `/home/wulf/Desktop/Benchmarking/flashnext-long-prefill-20261021/native-v2.cpp` | `4173ffc27e4728d841b60ce42f9fd56bfaaffb3980be04a089a534f5e9df97b8` |

## Other exact inventories

- `all-patches.json`: all 97 patch/diff paths, including upstream and legacy material, with targets and any original headers.
- `campaign-files.json`: 40376 retained code/build/evidence-document paths, with hashes; no generated caches.
- `runtime-helper-sources.json`: five handcrafted runtime diagnostic/helper Python files.
- `accepted-vllm-delta.json`: every accepted changed/added vLLM Python file and exact matching source snapshots.
- `accepted-files.json`: 3356 accepted regular package/frontend/shim files, excluding cache directories.
- `native-binary-matches.json`, `native-manifest-checks.json`: all 12 custom binaries, retained output paths and 81 checked source/recipe records.
- `native-dependency-trees.json`, `native-header-deltas.json`: complete native include/dependency file locks and base deltas.
- `patch-replay*.json`: original application order, outcomes and the generated-version-file exception.
- `image-lineage.json`: inspected image IDs and accepted-layer-prefix membership.

Paths beginning `/home/wulf/Desktop/Benchmarking/` refer to retained adjacent campaign evidence. The source-bearing external native variants are also copied under `originals/Benchmarking/`. Full runtime/tensor/request records stay in their original locations; this is not a duplicate of all raw evidence. The first raw GitHub releases API response was empty; use the independently saved `github-release-list.json` and per-release view receipt for published identities.
