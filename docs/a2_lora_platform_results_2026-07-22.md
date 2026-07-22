# A2 LoRA 骞冲彴娴嬭瘯缁撴灉锛?026-07-22锛?
鏈〉璁板綍 A2 LoRA 瀹归噺娑堣瀺鐨勬寮忓钩鍙版祴璇曠粨鏋溿€傚钩鍙板垎鏁板潎涓虹櫨鍒嗘瘮锛?锛夈€?
| 瀹為獙 | 鎺ㄧ悊鏂瑰紡 | 骞冲彴鍒嗘暟 | 鎻愪氦鍖?|
|---|---|---:|---|
| A2_LORA_MIN | 瑁告帹鐞?| **61.1167** | `outputs/a2_lora_min_knn/A2_LORA_MIN_KNN_DROP/seed42/submissions/submission.zip` |
| A2_LORA_MIN | horizontal_flip TTA | **61.6574** | `outputs/a2_lora_min_knn/A2_LORA_MIN_KNN_DROP/seed42/submissions_tta/submission.zip` |
| A2_LORA_FULL | 瑁告帹鐞?| **61.5733** | `outputs/a2_lora_full_knn/A2_LORA_FULL_KNN_DROP/seed42/submissions/submission.zip` |
| A2_LORA_FULL | horizontal_flip TTA | **62.1781** | `outputs/a2_lora_full_knn/A2_LORA_FULL_KNN_DROP/seed42/submissions_tta/submission.zip` |

## 缁撹

- 褰撳墠 A2 LoRA 娑堣瀺涓紝`A2_LORA_FULL + horizontal_flip TTA` 鏈€楂橈紝涓?**62.1781%**銆?- 鐩稿 `A2_LORA_MIN + horizontal_flip TTA`锛孎ULL 閰嶇疆鎻愬崌 **+0.5207 涓櫨鍒嗙偣**銆?- MIN 瑁告帹鐞嗕负 61.1167%锛屾瘮鏃㈡湁 A2 horizontal-flip 鍩虹嚎 61.2128% 浣?**0.0961 涓櫨鍒嗙偣**锛涘姞鍏?TTA 鍚庢彁鍗囧埌 61.6574%锛?*+0.4446 涓櫨鍒嗙偣**锛夈€?- FULL 瑁告帹鐞嗗钩鍙板垎鏁板凡纭 **61.5733%**锛涗笉鍐嶇敤鏈湴楠岃瘉鎸囨爣鏇夸唬骞冲彴缁撴灉銆?- 璇ョ粨鏋滀粛浣庝簬褰撳墠鐙珛瀹為獙璁板綍涓殑 F1+M1 缁勫悎 63.3276%锛屽洜姝?A2 LoRA 鐨勪富瑕佷环鍊兼槸楠岃瘉 LoRA 瀹归噺閰嶇疆锛岃€屼笉鏄彇浠ｅ綋鍓嶆渶浣虫彁浜ゃ€?
## 澶嶇幇涓庡悎瑙勮鏄?
鍥涗釜鎺ㄧ悊鍖呭潎浣跨敤鍗曟鏌ョ偣锛汿TA 鍖呴澶栧 horizontal flip 鍓嶅悜骞惰瀺鍚堟鐜囥€傝嫢姣旇禌瑙勫垯灏嗏€滃崟娆℃帹鐞嗏€濅弗鏍艰В閲婁负姣忓紶鍥剧墖鍙兘鍓嶅悜涓€娆★紝搴斾紭鍏堟彁浜よ８鎺ㄧ悊鍖咃紱鏈〉鍒嗘暟浠呰褰曞钩鍙板疄娴嬶紝涓嶆敼鍙樿鍒欏垽鏂€?