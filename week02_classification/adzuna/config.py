

UNK_IX, PAD_IX = 0, 1
TEXT_COLUMNS = ["Title", "FullDescription"]
CATEGORICAL_COLUMNS = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
COMPANY_COLUMN = 'Company'
TARGET_COLUMN = "Log1pSalary"
BATCH_SIZE = 16
EPOCHS = 5
TITLE_MAX_LEN=50
DESC_MAX_LEN=1550
