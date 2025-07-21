# ruff: noqa : E501
ElectronWP = {
    "Full2018v9": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'electron_col[LF_idx]["mvaFall17V2Iso_WPL"]',
                    ],
                },
            },
        },
        "TightObjWP": {
            "mvaFall17V2Iso_WP90": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP90"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
            "mvaFall17V2Iso_WP80": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP80"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
        },
    },
    "Full2017v9": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'electron_col[LF_idx]["mvaFall17V2Iso_WPL"]',
                    ],
                },
            },
        },
        "TightObjWP": {
            "mvaFall17V2Iso_WP90": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP90"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
            "mvaFall17V2Iso_WP80": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP80"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
        },
    },
    "Full2016v9HIPM": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'electron_col[LF_idx]["mvaFall17V2Iso_WPL"]',
                    ],
                },
            },
        },
        "TightObjWP": {
            "mvaFall17V2Iso_WP90": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP90"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
            "mvaFall17V2Iso_WP80": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP80"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
        },
    },
    "Full2016v9noHIPM": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'electron_col[LF_idx]["mvaFall17V2Iso_WPL"]',
                    ],
                },
            },
        },
        "TightObjWP": {
            "mvaFall17V2Iso_WP90": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP90"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
            "mvaFall17V2Iso_WP80": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(electron_col[LF_idx]["eta"]) < 2.5',
                        'electron_col[LF_idx]["mvaFall17V2Iso_WP80"]',
                    ],
                    # Barrel
                    'abs(electron_col[LF_idx]["eta"]) <= 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) < 1.44',
                    ],
                    # EndCap
                    'abs(electron_col[LF_idx]["eta"]) > 1.479': [
                        'abs(electron_col[LF_idx]["eta"]) > 1.57',
                    ],
                },
            },
        },
    },
}

MuonWP = {
    "Full2018v9": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["looseId"]',
                        #'muon_col[LF_idx]["pfRelIso04_all"] < 0.4',
                    ],
                },
            },
        },
        "TightObjWP": {
            "cut_mediumPromptId": { # mediumId + dz<0.1 + dxy<0.02
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
        },
    },
    "Full2017v9": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["looseId"]',
                        #'muon_col[LF_idx]["pfRelIso04_all"] < 0.4',
                    ],
                },
            },
        },
        "TightObjWP": {
            "cut_mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
        },
    },
    "Full2016v9HIPM": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["looseId"]',
                        #'muon_col[LF_idx]["pfRelIso04_all"] < 0.4',
                    ],
                },
            },
        },
        "TightObjWP": {
            "cut_mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
        },
    },
    "Full2016v9noHIPM": {
        "FakeObjWP": {
            "HLTsafe": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["looseId"]',
                        #'muon_col[LF_idx]["pfRelIso04_all"] < 0.4',
                    ],
                },
            },
        },
        "TightObjWP": {
            "cut_mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "cut_highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
        },
    },
}
