PAGE_WIDTH_PX = 1240
PAGE_HEIGHT_PX = 1754
PAGE_MARGIN_PX = 110
BODY_FONT_SIZE_PX = 26
TITLE_FONT_SIZE_PX = 44
ROW_HEIGHT_PX = 52
WHITE_LEVEL = 255

FONT_DIRECTORIES = ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts")
FONT_FILE_CANDIDATES = ("arial.ttf", "times.ttf", "cour.ttf", "calibri.ttf", "DejaVuSans.ttf")

DEFAULT_DOCUMENT_COUNT = 20
DEFAULT_RANDOM_SEED = 2026

BLUR_SIGMA_PX = 2.2
GAUSSIAN_NOISE_STD = 28.0
SALT_PEPPER_RATIO = 0.02
ILLUMINATION_MIN_FACTOR = 0.35
LOW_CONTRAST_RANGE = (140, 190)
SKEW_ANGLE_RANGE_DEG = (3.0, 8.0)

MEDIAN_DENOISE_KERNEL = 3
NOISE_SKIP_SIGMA = 2.0
NLM_SIGMA_SCALE = 1.0
NLM_MIN_STRENGTH = 3.0
NLM_MAX_STRENGTH = 25.0
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
ILLUMINATION_CLOSE_KERNEL = 41
ILLUMINATION_MEDIAN_KERNEL = 51
STRETCH_LOW_PERCENTILE = 0.05
STRETCH_HIGH_PERCENTILE = 50.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
DESKEW_SEARCH_RANGE_DEG = 12.0
DESKEW_COARSE_STEP_DEG = 0.5
DESKEW_FINE_STEP_DEG = 0.1
DESKEW_PREVIEW_SCALE = 0.5
DESKEW_MIN_CORRECTION_DEG = 0.3
PSNR_CAP_DB = 100.0
ADAPTIVE_BLOCK_SIZE = 35
ADAPTIVE_OFFSET = 15
MORPHOLOGY_KERNEL = (2, 2)

LINE_GROUP_HEIGHT_FRACTION = 0.6

CORE_ENHANCEMENT_STAGES = ["gray", "denoise", "illumination", "stretch", "deskew"]
BINARIZATION_STAGES = ["adaptiveBinarize", "morphology"]
PIPELINE_METHODS = {
    "gray": ["gray"],
    "otsu": ["gray", "otsuBinarize"],
    "enhanced": CORE_ENHANCEMENT_STAGES,
    "enhancedClahe": ["gray", "denoise", "illumination", "stretch", "clahe", "deskew"],
    "enhancedBinary": CORE_ENHANCEMENT_STAGES + BINARIZATION_STAGES,
}
ABLATION_STAGES = [
    "gray", "denoise", "illumination", "stretch", "clahe", "deskew",
    "adaptiveBinarize", "morphology",
]
ABLATION_CONDITION = "combined"
CONDITION_NAMES = ("clean", "blur", "noise", "unevenLight", "lowContrast", "skew", "combined")
