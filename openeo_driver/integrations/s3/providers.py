"""
This file ties regions to cloud providers.
"""

# Providers
CF = "cf"  # CloudFerro
OTC = "otc"  # Open Telekom cloud
EODATA = "eodata"  # eodata is considered a separate cloud provider stack
UNKNOWN = "unknown"  # unknown provider uses the legacy behavior


def get_s3_provider(region_name: str) -> str:
    region_name_lc = region_name.lower()
    if region_name_lc in ["waw3-1", "waw3-2", "waw4-1"]:
        return CF
    elif region_name_lc in ["eu-nl", "eu-de"]:
        return OTC
    elif region_name_lc in ["eodata"]:
        return EODATA
    else:
        return UNKNOWN
