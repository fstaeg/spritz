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
                    ],
                },
            },
        },
        "TightObjWP": {
            "mediumPromptId": { # mediumId + dz<0.1 + dxy<0.02
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                    ],
                },
            },
            "tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                    ],
                },
            },
            "highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15'
                    ],
                },
            },
            "RelIso": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "RelIso_loose": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.3',
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
                    ],
                },
            },
        },
        "TightObjWP": {
            "mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                    ],
                },
            },
            "tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                    ],
                },
            },
            "highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                    ],
                },
            },
            "RelIso": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "RelIso_loose": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.3',
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
                    ],
                },
            },
        },
        "TightObjWP": {
            "mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                    ],
                },
            },
            "tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                    ],
                },
            },
            "highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                    ],
                },
            },
            "RelIso": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "RelIso_loose": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.3',
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
                    ],
                },
            },
        },
        "TightObjWP": {
            "mediumPromptId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["mediumPromptId"]',
                    ],
                },
            },
            "tightId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["tightId"]',
                    ],
                },
            },
            "highPtId": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'abs(muon_col[LF_idx]["eta"]) < 2.4',
                        'muon_col[LF_idx]["highPtId"] == 2',
                    ],
                },
            },
            "RelIso": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.15',
                    ],
                },
            },
            "RelIso_loose": {
                "cuts": {
                    # Common cuts
                    "True": [
                        'muon_col[LF_idx]["pfRelIso04_all"] < 0.3',
                    ],
                },
            },
        },
    },
}
