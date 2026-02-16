import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Download and parse HURDAT2 data
class HURDAT2Loader:


    def load_data(self, filepath='hurdat2-1851-2024.txt'):

        storms = []
        current_storm = None

        with open(filepath, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',')]

                if parts[0].startswith('AL'):

                    current_storm = parts[0]
                else:

                    timestamp = parts[0]
                    lat = self.parse_latlon(parts[4])
                    lon = self.parse_latlon(parts[5])
                    vmax = float(parts[6]) if parts[6].strip() != '-999' else np.nan
                    pressure = float(parts[7]) if parts[7].strip() != '-999' else np.nan

                    storms.append({
                        'storm_id': current_storm,
                        'time': pd.to_datetime(timestamp),
                        'lat': lat,
                        'lon': lon,
                        'vmax_kt': vmax,
                        'pressure_mb': pressure
                    })

        df = pd.DataFrame(storms)
        df['vmax_ms'] = df['vmax_kt'] * 0.514444
        return df

    def parse_latlon(self, s):

        val = float(s[:-1])
        if s[-1] in ['S', 'W']:
            val = -val
        return val


# Create training dataset
class HurricaneDataset(Dataset):


    def __init__(self, hurdat_df, time_window=6):

        self.samples = []

        # Group by storm
        for storm_id, group in hurdat_df.groupby('storm_id'):
            group = group.sort_values('time').reset_index(drop=True)


            for i in range(len(group) - 1):
                current = group.iloc[i]
                future = group.iloc[i + 1]


                dt_hours = (future['time'] - current['time']).total_seconds() / 3600
                if dt_hours > 12:
                    continue

                # Compute target
                delta_lat = future['lat'] - current['lat']
                delta_lon = future['lon'] - current['lon']
                delta_vmax = future['vmax_ms'] - current['vmax_ms']


                features = np.array([
                    current['lat'],
                    current['lon'],
                    current['vmax_ms'],
                    current['pressure_mb'],
                    delta_lat / dt_hours,  # current velocity
                    delta_lon / dt_hours
                ])

                self.samples.append({
                    'features': torch.tensor(features, dtype=torch.float32),
                    'target': torch.tensor([delta_lat, delta_lon, delta_vmax], dtype=torch.float32)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['features'], self.samples[idx]['target']



def train_cnn_on_historical_data():
    """Complete training pipeline."""

    print("=" * 70)
    print("Training CNN on HURDAT2 database")
    print("=" * 70)

    # Load data
    print("\n1. Loading HURDAT2 data...")
    loader = HURDAT2Loader()
    df = loader.load_data('hurdat2-1851-2024.txt')
    print(f"   Loaded {len(df)} observations from {df['storm_id'].nunique()} storms")


    print("\n2. Creating training dataset...")
    dataset = HurricaneDataset(df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"   Train: {train_size} samples, Val: {val_size} samples")


    print("\n3. Initializing model...")
    model = nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train
    print("\n4. Training...")
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {train_loss / len(train_loader):.6f} | "
                  f"Val Loss: {val_loss / len(val_loader):.6f}")

    # Save model
    print("\n5. Saving model...")
    torch.save(model.state_dict(), 'hurricane_cnn_weights.pth')
    print(" Model saved to hurricane_cnn_weights.pth")

    return model