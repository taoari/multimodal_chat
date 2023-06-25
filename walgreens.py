WALGREENS_CATALOG = None


def import_walgreens_catalog():
    global WALGREENS_CATALOG
    if WALGREENS_CATALOG is None:
        import pandas as pd
        df = pd.read_csv('data/walgreens_full_import_passthru.csv')
        WALGREENS_CATALOG = df


def walgreens_search_by_barcode(barcode):
    if WALGREENS_CATALOG is None:
        import_walgreens_catalog()
    
    df = WALGREENS_CATALOG
    df_match = df.loc[df['gtin'].astype(int) == int(barcode)]
    return df_match if len(df_match) > 0 else None