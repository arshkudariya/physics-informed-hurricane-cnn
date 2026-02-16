# train_cnn.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


print("CNN Training Script")



class HURDAT2Loader:
    """Parse HURDAT2 hurricane database."""

    def load_data(self, filepath='hurdat2-1851-2024.txt'):
        print(f"\nLoading data from: {filepath}")
        storms = []
        current_storm = None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = [p.strip() for p in line.split(',')]

                    if parts[0].startswith('AL') or parts[0].startswith('CP'):
                        current_storm = parts[0]
                    else:
                        try:
                            timestamp = parts[0]
                            lat = self.parse_latlon(parts[4])
                            lon = self.parse_latlon(parts[5])
                            vmax = float(parts[6]) if parts[6].strip() not in ['-999', ''] else np.nan
                            pressure = float(parts[7]) if parts[7].strip() not in ['-999', ''] else np.nan


                            if np.isnan(lat) or np.isnan(lon) or np.isnan(vmax):
                                continue

                            storms.append({
                                'storm_id': current_storm,
                                'time': pd.to_datetime(timestamp, format='%Y%m%d%H'),
                                'lat': lat,
                                'lon': lon,
                                'vmax_kt': vmax,
                                'pressure_mb': pressure if not np.isnan(pressure) else 1013.0
                            })
                        except:
                            continue

            df = pd.DataFrame(storms)
            df['vmax_ms'] = df['vmax_kt'] * 0.514444


            df = df.dropna(subset=['lat', 'lon', 'vmax_ms', 'pressure_mb'])

            # Remove outliers
            df = df[df['lat'].between(-90, 90)]
            df = df[df['lon'].between(-180, 180)]
            df = df[df['vmax_ms'].between(0, 100)]
            df = df[df['pressure_mb'].between(850, 1050)]

            print(f"✓ Loaded {len(df)} observations from {df['storm_id'].nunique()} storms")
            return df

        except FileNotFoundError:
            print(f"Error: Could not find '{filepath}'")
            exit(1)

    def parse_latlon(self, s):
        """Parse '25.3N' or '75.2W' format."""
        s = s.strip()
        if not s or s == '-999':
            return np.nan
        val = float(s[:-1])
        if s[-1] in ['S', 'W']:
            val = -val
        return val


class HurricaneDataset(Dataset):
    """PyTorch dataset for hurricane prediction."""

    def __init__(self, hurdat_df):
        print("\nCreating training samples...")
        self.samples = []

        for storm_id, group in hurdat_df.groupby('storm_id'):
            group = group.sort_values('time').reset_index(drop=True)

            if len(group) < 2:
                continue

            for i in range(len(group) - 1):
                current = group.iloc[i]
                future = group.iloc[i + 1]

                dt_hours = (future['time'] - current['time']).total_seconds() / 3600
                if dt_hours > 12 or dt_hours < 1:
                    continue


                features = [
                    current['lat'] / 90.0,
                    current['lon'] / 180.0,
                    current['vmax_ms'] / 80.0,
                    current['pressure_mb'] / 1013.0,
                    dt_hours / 12.0
                ]


                target = [
                    (future['lat'] - current['lat']) / 5.0,  # Scale down
                    (future['lon'] - current['lon']) / 5.0,
                    (future['vmax_ms'] - current['vmax_ms']) / 20.0
                ]


                if any(np.isnan(features)) or any(np.isnan(target)):
                    continue

                # Check for extreme values
                if max(abs(x) for x in features) > 10 or max(abs(x) for x in target) > 10:
                    continue

                self.samples.append({
                    'features': torch.tensor(features, dtype=torch.float32),
                    'target': torch.tensor(target, dtype=torch.float32)
                })

        print(f"Created {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['features'], self.samples[idx]['target']


class SimpleHurricaneCNN(nn.Module):


    def __init__(self):
        super(SimpleHurricaneCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 3)
        )

        # Initialize weights properly, **make sure to access properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def train_model():



    loader = HURDAT2Loader()
    df = loader.load_data('hurdat2-1851-2024.txt')


    dataset = HurricaneDataset(df)

    if len(dataset) < 100:
        print(" Error: Not enough training samples")
        exit(1)


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print(f"\nTrain: {train_size}, Val: {val_size}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = SimpleHurricaneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
    criterion = nn.MSELoss()


    print("\nTraining...")
    print("-" * 70)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(50):

        model.train()
        train_loss = 0.0
        n_batches = 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)


            if torch.isnan(features).any() or torch.isnan(targets).any():
                continue

            optimizer.zero_grad()
            predictions = model(features)


            if torch.isnan(predictions).any():
                print(f"Warning: NaN in predictions at epoch {epoch}")
                continue

            loss = criterion(predictions, targets)

            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch}")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print("Error: No valid batches")
            break

        train_loss = train_loss / n_batches


        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)

                if torch.isnan(features).any() or torch.isnan(targets).any():
                    continue

                predictions = model(features)

                if torch.isnan(predictions).any():
                    continue

                loss = criterion(predictions, targets)

                if torch.isnan(loss):
                    continue

                val_loss += loss.item()
                n_val_batches += 1

        if n_val_batches > 0:
            val_loss = val_loss / n_val_batches


        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/50 | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), 'hurricane_cnn_weights.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break


    print("Training complete")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(" Model saved to: hurricane_cnn_weights.pth")



if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()