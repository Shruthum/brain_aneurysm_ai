ANATOMY_LABELS = [
    "Other Posterior Circulation",
    "Basilar Tip",
    "Right Posterior Communicating Artery",
    "Left Posterior Communicating Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Infraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Middle Cerebral Artery",
    "Left Middle Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Anterior Cerebral Artery",
    "Anterior Communicating Artery",
]

ANATOMY_TO_INDEX = { name: i for i, name in enumerate(ANATOMY_LABELS) }

INDEX_TO_ANATOMY = { i: name for i, name in enumerate(ANATOMY_LABELS) }

RL_ACTIONS = {
    0: "POS_X",
    1: "NEG_X",
    2: "POS_Y",
    3: "NEG_Y",
    4: "POS_Z",
    5: "NEG_Z",
    6: "ZOOM_IN",
    7: "ZOOM_OUT",
    8: "STOP",
}
def anatomy_to_id(name):
    if name not in ANATOMY_TO_INDEX:
        raise ValueError(
            f"Unknown anatomy label: {name}"
        )
    return ANATOMY_TO_INDEX[name]