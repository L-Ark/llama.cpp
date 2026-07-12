{
  "allowed_source_idx": [],
  "batch_summary": {
    "candidate_pct_hist": {
      "25-50": 2954,
      "50-75": 1803,
      "<25": 1638,
      ">=75": 459
    },
    "candidate_rows_hist": {
      "1": 3427,
      "2": 1700,
      "3": 891,
      "4": 500,
      "5": 222,
      "6": 90,
      "7": 22,
      "8": 2
    },
    "hit_batches": 6854,
    "hit_candidate_bytes": 80078831616,
    "hit_candidate_gib": 74.5792236328125,
    "ram_dominant_batches": 226,
    "ram_dominant_candidate_bytes": 7864041472,
    "ram_dominant_candidate_gib": 7.3239593505859375,
    "ram_only_batches": 154,
    "ram_only_candidate_bytes": 2164162560,
    "ram_only_candidate_gib": 2.015533447265625,
    "selected_id_count": 2159,
    "total_batches": 11134
  },
  "budget_mib": 12288,
  "exclude_vram": false,
  "kind": "kimi_ram_candidate_multidev_screen",
  "max_jobs": 8,
  "metadata": {
    "counters": {
      "rows_above_max_jobs": 24243,
      "rows_total": 75839
    },
    "source_bytes": {
      "0": 244537180160,
      "1": 4945346560,
      "2": 4181606400,
      "3": 5525114880,
      "4": 5444911104,
      "5": 5411368960,
      "6": 5843083264,
      "7": 6248722432,
      "8": 6933405696,
      "9": 6696259584,
      "10": 2212265984
    },
    "source_rows": {
      "0": 42384,
      "1": 784,
      "2": 692,
      "3": 957,
      "4": 920,
      "5": 935,
      "6": 1054,
      "7": 1107,
      "8": 1246,
      "9": 1141,
      "10": 376
    },
    "vram_overlap_bytes": 0,
    "vram_overlap_rows": 0
  },
  "min_count": 2,
  "min_prompts": 2,
  "prompts": [
    "dev_france_regression",
    "dev_intelligence_general"
  ],
  "roles": [
    "down",
    "gate",
    "up"
  ],
  "route_profiles": [
    "/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538/dev_france_regression/route-profile.csv",
    "/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538/dev_intelligence_general/route-profile.csv"
  ],
  "selected_bytes": 12884049920,
  "selected_entries": 2159,
  "selected_mib": 12287.1875,
  "selected_source_bytes": {
    "0": 78142324736,
    "1": 1936506880
  },
  "selected_source_rows": {
    "0": 13013,
    "1": 307
  },
  "selected_weighted_bytes": 80078831616,
  "selected_weighted_gib": 74.5792236328125,
  "top_entries": [
    {
      "bytes": 171573248,
      "expert_bytes": 7798784,
      "expert_idx": 26,
      "layer": 4,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 1,
      "role": "down",
      "rows": 22,
      "source_bytes": {
        "0": 171573248
      },
      "sources": {
        "0": 22
      },
      "tensor": "blk.4.ffn_down_exps.weight"
    },
    {
      "bytes": 126156800,
      "expert_bytes": 6307840,
      "expert_idx": 116,
      "layer": 1,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 2,
      "role": "down",
      "rows": 20,
      "source_bytes": {
        "1": 126156800
      },
      "sources": {
        "1": 20
      },
      "tensor": "blk.1.ffn_down_exps.weight"
    },
    {
      "bytes": 156893184,
      "expert_bytes": 8257536,
      "expert_idx": 84,
      "layer": 6,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 3,
      "role": "down",
      "rows": 19,
      "source_bytes": {
        "0": 156893184
      },
      "sources": {
        "0": 19
      },
      "tensor": "blk.6.ffn_down_exps.weight"
    },
    {
      "bytes": 156893184,
      "expert_bytes": 8257536,
      "expert_idx": 84,
      "layer": 7,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 4,
      "role": "down",
      "rows": 19,
      "source_bytes": {
        "0": 156893184
      },
      "sources": {
        "0": 19
      },
      "tensor": "blk.7.ffn_down_exps.weight"
    },
    {
      "bytes": 156893184,
      "expert_bytes": 8257536,
      "expert_idx": 235,
      "layer": 7,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 5,
      "role": "down",
      "rows": 19,
      "source_bytes": {
        "0": 156893184
      },
      "sources": {
        "0": 19
      },
      "tensor": "blk.7.ffn_down_exps.weight"
    },
    {
      "bytes": 140378112,
      "expert_bytes": 7798784,
      "expert_idx": 103,
      "layer": 20,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 6,
      "role": "down",
      "rows": 18,
      "source_bytes": {
        "0": 140378112
      },
      "sources": {
        "0": 18
      },
      "tensor": "blk.20.ffn_down_exps.weight"
    },
    {
      "bytes": 140378112,
      "expert_bytes": 7798784,
      "expert_idx": 67,
      "layer": 22,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 7,
      "role": "down",
      "rows": 18,
      "source_bytes": {
        "0": 140378112
      },
      "sources": {
        "0": 18
      },
      "tensor": "blk.22.ffn_down_exps.weight"
    },
    {
      "bytes": 140378112,
      "expert_bytes": 7798784,
      "expert_idx": 246,
      "layer": 23,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 8,
      "role": "down",
      "rows": 18,
      "source_bytes": {
        "0": 140378112
      },
      "sources": {
        "0": 18
      },
      "tensor": "blk.23.ffn_down_exps.weight"
    },
    {
      "bytes": 113541120,
      "expert_bytes": 6307840,
      "expert_idx": 270,
      "layer": 13,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 9,
      "role": "down",
      "rows": 18,
      "source_bytes": {
        "0": 113541120
      },
      "sources": {
        "0": 18
      },
      "tensor": "blk.13.ffn_down_exps.weight"
    },
    {
      "bytes": 113541120,
      "expert_bytes": 6307840,
      "expert_idx": 65,
      "layer": 11,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 10,
      "role": "down",
      "rows": 18,
      "source_bytes": {
        "0": 113541120
      },
      "sources": {
        "0": 18
      },
      "tensor": "blk.11.ffn_down_exps.weight"
    },
    {
      "bytes": 140378112,
      "expert_bytes": 8257536,
      "expert_idx": 288,
      "layer": 7,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 11,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 140378112
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.7.ffn_down_exps.weight"
    },
    {
      "bytes": 140378112,
      "expert_bytes": 8257536,
      "expert_idx": 10,
      "layer": 18,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 12,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 140378112
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.18.ffn_down_exps.weight"
    },
    {
      "bytes": 132579328,
      "expert_bytes": 7798784,
      "expert_idx": 83,
      "layer": 21,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 13,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 132579328
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.21.ffn_down_exps.weight"
    },
    {
      "bytes": 132579328,
      "expert_bytes": 7798784,
      "expert_idx": 213,
      "layer": 26,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 14,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 132579328
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.26.ffn_down_exps.weight"
    },
    {
      "bytes": 107233280,
      "expert_bytes": 6307840,
      "expert_idx": 76,
      "layer": 27,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 15,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 107233280
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.27.ffn_down_exps.weight"
    },
    {
      "bytes": 107233280,
      "expert_bytes": 6307840,
      "expert_idx": 363,
      "layer": 29,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 16,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 107233280
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.29.ffn_down_exps.weight"
    },
    {
      "bytes": 107233280,
      "expert_bytes": 6307840,
      "expert_idx": 353,
      "layer": 32,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 17,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 107233280
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.32.ffn_down_exps.weight"
    },
    {
      "bytes": 107233280,
      "expert_bytes": 6307840,
      "expert_idx": 286,
      "layer": 14,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 18,
      "role": "down",
      "rows": 17,
      "source_bytes": {
        "0": 107233280
      },
      "sources": {
        "0": 17
      },
      "tensor": "blk.14.ffn_down_exps.weight"
    },
    {
      "bytes": 132120576,
      "expert_bytes": 8257536,
      "expert_idx": 153,
      "layer": 10,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 19,
      "role": "down",
      "rows": 16,
      "source_bytes": {
        "0": 132120576
      },
      "sources": {
        "0": 16
      },
      "tensor": "blk.10.ffn_down_exps.weight"
    },
    {
      "bytes": 132120576,
      "expert_bytes": 8257536,
      "expert_idx": 34,
      "layer": 6,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 20,
      "role": "down",
      "rows": 16,
      "source_bytes": {
        "0": 132120576
      },
      "sources": {
        "0": 16
      },
      "tensor": "blk.6.ffn_down_exps.weight"
    },
    {
      "bytes": 124780544,
      "expert_bytes": 7798784,
      "expert_idx": 340,
      "layer": 25,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 21,
      "role": "down",
      "rows": 16,
      "source_bytes": {
        "0": 124780544
      },
      "sources": {
        "0": 16
      },
      "tensor": "blk.25.ffn_down_exps.weight"
    },
    {
      "bytes": 100925440,
      "expert_bytes": 6307840,
      "expert_idx": 41,
      "layer": 1,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 22,
      "role": "down",
      "rows": 16,
      "source_bytes": {
        "1": 100925440
      },
      "sources": {
        "1": 16
      },
      "tensor": "blk.1.ffn_down_exps.weight"
    },
    {
      "bytes": 123863040,
      "expert_bytes": 8257536,
      "expert_idx": 356,
      "layer": 6,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 23,
      "role": "down",
      "rows": 15,
      "source_bytes": {
        "0": 123863040
      },
      "sources": {
        "0": 15
      },
      "tensor": "blk.6.ffn_down_exps.weight"
    },
    {
      "bytes": 116981760,
      "expert_bytes": 7798784,
      "expert_idx": 105,
      "layer": 23,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 24,
      "role": "down",
      "rows": 15,
      "source_bytes": {
        "0": 116981760
      },
      "sources": {
        "0": 15
      },
      "tensor": "blk.23.ffn_down_exps.weight"
    },
    {
      "bytes": 116981760,
      "expert_bytes": 7798784,
      "expert_idx": 107,
      "layer": 16,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 25,
      "role": "down",
      "rows": 15,
      "source_bytes": {
        "0": 116981760
      },
      "sources": {
        "0": 15
      },
      "tensor": "blk.16.ffn_down_exps.weight"
    },
    {
      "bytes": 94617600,
      "expert_bytes": 6307840,
      "expert_idx": 11,
      "layer": 56,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 26,
      "role": "down",
      "rows": 15,
      "source_bytes": {
        "0": 94617600
      },
      "sources": {
        "0": 15
      },
      "tensor": "blk.56.ffn_down_exps.weight"
    },
    {
      "bytes": 94617600,
      "expert_bytes": 6307840,
      "expert_idx": 305,
      "layer": 29,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 27,
      "role": "down",
      "rows": 15,
      "source_bytes": {
        "0": 94617600
      },
      "sources": {
        "0": 15
      },
      "tensor": "blk.29.ffn_down_exps.weight"
    },
    {
      "bytes": 115605504,
      "expert_bytes": 8257536,
      "expert_idx": 14,
      "layer": 7,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 28,
      "role": "down",
      "rows": 14,
      "source_bytes": {
        "0": 115605504
      },
      "sources": {
        "0": 14
      },
      "tensor": "blk.7.ffn_down_exps.weight"
    },
    {
      "bytes": 88309760,
      "expert_bytes": 6307840,
      "expert_idx": 173,
      "layer": 11,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 29,
      "role": "down",
      "rows": 14,
      "source_bytes": {
        "0": 88309760
      },
      "sources": {
        "0": 14
      },
      "tensor": "blk.11.ffn_down_exps.weight"
    },
    {
      "bytes": 88309760,
      "expert_bytes": 6307840,
      "expert_idx": 37,
      "layer": 32,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 30,
      "role": "down",
      "rows": 14,
      "source_bytes": {
        "0": 88309760
      },
      "sources": {
        "0": 14
      },
      "tensor": "blk.32.ffn_down_exps.weight"
    },
    {
      "bytes": 88309760,
      "expert_bytes": 6307840,
      "expert_idx": 31,
      "layer": 13,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 31,
      "role": "down",
      "rows": 14,
      "source_bytes": {
        "0": 88309760
      },
      "sources": {
        "0": 14
      },
      "tensor": "blk.13.ffn_down_exps.weight"
    },
    {
      "bytes": 88309760,
      "expert_bytes": 6307840,
      "expert_idx": 298,
      "layer": 2,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 32,
      "role": "down",
      "rows": 14,
      "source_bytes": {
        "1": 88309760
      },
      "sources": {
        "1": 14
      },
      "tensor": "blk.2.ffn_down_exps.weight"
    },
    {
      "bytes": 107347968,
      "expert_bytes": 8257536,
      "expert_idx": 155,
      "layer": 8,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 33,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 107347968
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.8.ffn_down_exps.weight"
    },
    {
      "bytes": 107347968,
      "expert_bytes": 8257536,
      "expert_idx": 198,
      "layer": 10,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 34,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 107347968
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.10.ffn_down_exps.weight"
    },
    {
      "bytes": 101384192,
      "expert_bytes": 7798784,
      "expert_idx": 79,
      "layer": 4,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 35,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 101384192
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.4.ffn_down_exps.weight"
    },
    {
      "bytes": 101384192,
      "expert_bytes": 7798784,
      "expert_idx": 331,
      "layer": 19,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 36,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 101384192
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.19.ffn_down_exps.weight"
    },
    {
      "bytes": 101384192,
      "expert_bytes": 7798784,
      "expert_idx": 197,
      "layer": 21,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 37,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 101384192
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.21.ffn_down_exps.weight"
    },
    {
      "bytes": 101384192,
      "expert_bytes": 7798784,
      "expert_idx": 344,
      "layer": 24,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 38,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 101384192
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.24.ffn_down_exps.weight"
    },
    {
      "bytes": 101384192,
      "expert_bytes": 7798784,
      "expert_idx": 154,
      "layer": 24,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 39,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 101384192
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.24.ffn_down_exps.weight"
    },
    {
      "bytes": 82001920,
      "expert_bytes": 6307840,
      "expert_idx": 64,
      "layer": 13,
      "prompts": [
        "dev_france_regression",
        "dev_intelligence_general"
      ],
      "rank": 40,
      "role": "down",
      "rows": 13,
      "source_bytes": {
        "0": 82001920
      },
      "sources": {
        "0": 13
      },
      "tensor": "blk.13.ffn_down_exps.weight"
    }
  ],
  "traces": [
    "/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538/dev_france_regression/io-read-trace.csv",
    "/root/lfz/runs/vendor-kimi-token-rate/20260712-active-goal-c7b69-io-read-trace-n32-114538/dev_intelligence_general/io-read-trace.csv"
  ],
  "vram_ids": 0
}
