import openassetpricing as oap
import time

def test_google_drive_connection():
    print("Testing connection to Open Asset Pricing data on Google Drive...")
    
    # List available release versions
    print("\nAvailable release versions:")
    oap.list_release()
    
    # Initialize OpenAP with the latest release
    print("\nInitializing OpenAP with the latest release...")
    start_time = time.time()
    openap = oap.OpenAP()
    end_time = time.time()
    print(f"Connection established in {end_time - start_time:.2f} seconds")
    
    # List available portfolios
    print("\nAvailable portfolios:")
    openap.list_port()
    
    # Download a small sample of data to verify connection
    print("\nDownloading a sample of data (BM predictor)...")
    start_time = time.time()
    df = openap.dl_port('op', 'polars', ['BM'])
    end_time = time.time()
    print(f"Data downloaded in {end_time - start_time:.2f} seconds")
    
    # Show a small sample of the data
    print("\nSample of downloaded data:")
    print(df.head(5))
    
    print("\nConnection test completed successfully!")

if __name__ == "__main__":
    test_google_drive_connection()
