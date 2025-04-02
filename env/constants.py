import numpy as np
import math

WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])

PP_SHIFT_Y = 0.168
GRASP_WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.336, 0.000], [-0.0001, 0.4]])
PLACE_WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.000, 0.336], [-0.0001, 0.4]])
PP_WORKSPACE_LIMITS = np.asarray([[0.164, 0.836], [-0.336, 0.336], [-0.0001, 0.4]])

# grasp lang
LANG_TEMPLATES = ["give me the {keyword}", # label
                "I need a {keyword}", # general label
                "grasp a {keyword} object", # shape or color
                "I want a {keyword} object", # shape or color
                "get something to {keyword}", # function
                ]

# place lang
PLACE_LANG_TEMPLATES = [ "put it {direction} the {reference}", # 1 obj
                         "place this {direction} the {reference}",
                         "move the object {direction} the {reference}"]

OBJ_DIRECTION = ["on", "in", "to", "near", "around", "next to"]
IN_REGION = ["on", "in", "to", "upside", "on top of"]
OUT_REGION = ["near", "around", "next to", "beside", "close to", "surrounding to"]

OBJ_DIRECTION_MAP = {
    "on": ["006", "007", "022", "074"],
    "in": ["006", "007", "022", "074"],
    "to": ["006", "007", "022", "074"],
    "near": ["002", "005", "006", "007", "008", "009", "011", "012", "013",
            "014", "015", "016", "017", "018", "020", "021", "022", "024",
            "026", "027", "028", "029", "030", "031", "032", "034", "037",
            "038", "039", "040", "041", "042", "043", "044", "045", "047", 
            "050", "052", "053", "055", "057", "058", "059", "061", "062", 
            "064", "066", "067", "068", "070", "072", "073", "074", "075",
            "076", "077", "078", "079", "080", "081", "082", "083", "084", 
            "085", "086", "087"],
    "around": ["002", "005", "006", "007", "008", "009", "011", "012", "013",
            "014", "015", "016", "017", "018", "020", "021", "022", "024",
            "026", "027", "028", "029", "030", "031", "032", "034", "037",
            "038", "039", "040", "041", "042", "043", "044", "045", "047", 
            "050", "052", "053", "055", "057", "058", "059", "061", "062", 
            "064", "066", "067", "068", "070", "072", "073", "074", "075",
            "076", "077", "078", "079", "080", "081", "082", "083", "084", 
            "085", "086", "087"],
    "next to": ["002", "005", "006", "007", "008", "009", "011", "012", "013",
            "014", "015", "016", "017", "018", "020", "021", "022", "024",
            "026", "027", "028", "029", "030", "031", "032", "034", "037",
            "038", "039", "040", "041", "042", "043", "044", "045", "047", 
            "050", "052", "053", "055", "057", "058", "059", "061", "062", 
            "064", "066", "067", "068", "070", "072", "073", "074", "075",
            "076", "077", "078", "079", "080", "081", "082", "083", "084", 
            "085", "086", "087"]   
    # close to, beside, by, adjacent to, near to, next to, around, surrounding, surrounding to, surrounding by, surrounding with, surrounding at, surrounding in, surrounding on, surrounding near, surrounding beside, surrounding next to, surrounding around, surrounding close to, surrounding adjacent to 
}

OBJ_UNSEEN_DIRECTION_MAP = {
    "on": ["gelatin_box", "pink_tea_box", "soap_dish", "yellow_bowl", "yellow_cup"],
    "in": ["soap_dish", "yellow_bowl", "yellow_cup"],
    "to": ["gelatin_box", "pink_tea_box", "soap_dish", "yellow_bowl", "yellow_cup"],
    "upside": ["gelatin_box", "pink_tea_box", "soap_dish", "yellow_bowl", "yellow_cup"],
    "on top of": ["gelatin_box", "pink_tea_box", "soap_dish", "yellow_bowl", "yellow_cup"],
    "near": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"],
    "around": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"],
    "next to": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"], 
    "beside": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"],
    "close to": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"], 
    "surrounding to": ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"],   
    # close to, beside, by, adjacent to, near to, next to, around, surrounding, surrounding to, surrounding by, surrounding with, surrounding at, surrounding in, surrounding on, surrounding near, surrounding beside, surrounding next to, surrounding around, surrounding close to, surrounding adjacent to 
}
MIN_DIS = 0.05 
SINGLE_REF_DISTANCE = 0.2
MAX_DIS = 0.15


# object
# Note that there are some multi-scale inbalanced grasp detection problem
# delete unsuitable objects e.g. head_shoulder
LABEL = ["tomato soup can", "banana", "red mug", "power drill", "strawberry", "apple", "lemon", 
        "peach", "pear", "orange", "knife", "flat screwdriver", "racquetball", "cup", "toy airplane",
        "dabao sod", "toothpaste", "darlie box", "dabao facewash", "pantene", "tape"]

GENERAL_LABEL = ["fruit", "container", "toy", "cup"]
COLOR_SHAPE = ["yellow", "red", "round"]
# loc = ["center", "bottom right", "center right", "top right", "bottom left", "center left", "top left"]
FUNCTION = ["eat", "drink", "play", "hold other things"]

LABEL_DIR_MAP = ["002", "005", "007", "008", "011", "012", "013",
                "014", "015", "016", "018", "020", "021", "022", "024", 
                "038", "041", "058", "061", "062", "066", "070"]
  
ALL_LABEL = ["tomato soup can", "banana", "red bowl", "red mug", "power drill", "scissors", "strawberry", "apple", "lemon", 
            "peach", "pear", "orange", "plum", "knife", "flat screwdriver", "racquetball", "cup", "toy airplane",
            "dabao sod", "box", "toothpaste", "white mouse", "darlie box", "soap", "dabao facewash", "pantene", "thera med", 
            "dove", "head shoulders care", "lion", "tape"]

ALL_LABEL_DIR_MAP = ["002", "005", "006", "007", "008", "009", "011", "012", "013",
                    "014", "015", "016", "017", "018", "020", "021", "022", "024",  
                    "038", "039", "041", "047", "058", "059", "061", "062", "064", 
                    "065", "066", "067", "070"]

KEYWORD_DIR_MAP = {"fruit": ["005", "011", "012", "013", "014", "015", "016", "017"],
                    "container": ["006", "007", "022"],
                    "toy": ["024", "026", "027", "028", "029", "030", "031",
                            "075", "076", "077", "078", "079", "080", "081", "082", "083", 
                            "084", "085", "086", "087"],
                    "cup": ["022"],
                    "ball": ["021"],
                    "yellow": ["005", "013", "028", "031"],
                    "red": ["011", "012"],
                    "round": ["016", "017", "021"],
                    "box": ["039"],
                    "eat": ["005", "011", "012", "013", "014", "015", "016", "017"], 
                    "drink": ["057"],
                    "play": ["024", "026", "027", "028", "029", "030", "031"],
                    "hold other things": ["006", "007", "022"]}

UNSEEN_LABEL = ["black marker", "bleach cleanser", "blue moon", "gelatin box", "magic clean", "pink tea box", "red marker", 
                "remote controller", "repellent", "shampoo", "small clamp", "soap dish", "sugar", "sugar", "two color hammer",
                "yellow bowl", "yellow cup"]

UNSEEN_LABEL_DIR_MAP = ["black_marker", "bleach_cleanser", "blue_moon", "gelatin_box", "magic_clean", "pink_tea_box", "red_marker", 
                "remote_controller_1", "repellent", "shampoo", "small_clamp", "soap_dish", "suger_1", "suger_2", "two_color_hammer",
                "yellow_bowl", "yellow_cup"]

UNSEEN_GENERAL_LABEL = ["sugar", "container"]
UNSEEN_COLOR_SHAPE = ["yellow", "red"]
UNSEEN_FUNCTION = ["clean"]
UNSEEN_KEYWORD_DIR_MAP = {"sugar": ["suger_1", "suger_2"],
                    "container": ["soap_dish", "yellow_cup"],
                    "yellow": ["yellow_bowl", "yellow_cup"],
                    "red": ["red_marker"],
                    "clean": ["bleach_cleanser", "blue_moon", "magic_clean", "shampoo"]}


UNSEEN_KEYWORD_DIR_MAP_PLACE = {"container": ["soap_dish", "yellow_cup"]}

# image
PP_PIXEL_SIZE = 0.003
PIXEL_SIZE = 0.002
IMAGE_SIZE = 224
