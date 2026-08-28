Trigger = {
    # --------------------------- Full2018v9---------------------------------
    "Full2018v9": {
        # Run A-B (before HEM15/16 issue)
        1: {
            "begin": 315257,
            "end": 318944,
            "lumi": 20.973214715,
            "EMTFBug": False,
            "HEMIssue": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
                "SingleMu": ["HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele32_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
                "SingleMu": ["HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele32_WPTight_Gsf"],
            },
        },
        # Run B-D (after HEM15/16 issue)
        2: {
            "begin": 319077,
            "end": 325175,
            "lumi": 38.588047804,
            "EMTFBug": False,
            "HEMIssue": True,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
                "SingleMu": ["HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele32_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
                "SingleMu": ["HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele32_WPTight_Gsf"],
            },
        },
    },
    # --------------------------- Full2017v9---------------------------------
    "Full2017v9": {
        # Run B
        1: {
            "begin": 297047,
            "end": 299329,
            "lumi": 4.880866827,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"],
                "SingleMu": ["HLT_IsoMu27"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele35_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"],
                "SingleMu": ["HLT_IsoMu27"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele35_WPTight_Gsf"],
            },
        },
        # Run C->F
        2: {
            "begin": 299368,
            "end": 306462,
            "lumi": 37.187361834,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"],
                "SingleMu": ["HLT_IsoMu27"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele35_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"],
                "SingleMu": ["HLT_IsoMu27"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"],
                "SingleEle": ["HLT_Ele35_WPTight_Gsf"],
            },
        },
    },
    # --------------------------- Full2016v9HIPM---------------------------------
    "Full2016v9HIPM": {
        # Run B->F: no DZ filters on e-mu + HIPM problem
        1: {
            "begin": 272760,
            "end": 278240,
            "lumi": 17.846373587,
            "EMTFBug": True,
            "DATA": {
                "EleMu": [
                    "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
        },
        # Run F: DZ filters on e-mu + HIPM problem
        2: {
            "runList": [
                278273, 278274, 278288, 278289, 278290, 278308, 278309, 278310, 278315, 278345, 278346, 278349, 278366, 278406, 278509, 278761, 278770, 278806, 278807
            ],
            "lumi": 1.655228035,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
        },
    },
    # --------------------------- Full2016v9noHIPM---------------------------------
    "Full2016v9noHIPM": {
        # Run F: DZ filters on e-mu
        3: {
            "runList": [
                278769, 278801, 278802, 278803, 278804, 278805, 278808
            ],
            "lumi": 4.18771191,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
        },
        # Run G: No change of trigger
        4: {
            "begin": 278820,
            "end": 280385,
            "lumi": 7.653261227,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
        },
        # Run H: Switch to DZ version of Double Mu triggers
        5: {
            "begin": 281613,
            "end": 284044,
            "lumi": 8.740119304,
            "EMTFBug": False,
            "DATA": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
            "MC": {
                "EleMu": [
                    "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
                    "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
                ],
                "DoubleMu": [
                    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
                ],
                "SingleMu": ["HLT_IsoTkMu24", "HLT_IsoMu24"],
                "DoubleEle": ["HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
                "SingleEle": ["HLT_Ele27_WPTight_Gsf", "HLT_Ele25_eta2p1_WPTight_Gsf"],
            },
        },
    },
}
