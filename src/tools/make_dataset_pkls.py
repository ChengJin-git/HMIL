import pickle
import csv
import os
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset splits for MIL training')
    parser.add_argument('--feature_directory', type=str, required=True,
                      help='Path to the feature files directory')
    parser.add_argument('--root', type=str, required=True,
                      help='Output directory for the generated splits')
    parser.add_argument('--label_csv', type=str, required=True,
                      help='Path to the CSV file containing labels')
    parser.add_argument('--split_csv_root', type=str, required=True,
                      help='Path to the directory containing split CSV files')
    parser.add_argument('--folds', type=int, default=10,
                      help='Number of folds for cross-validation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.root).mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    for fold in range(args.folds):
        train_splits = []
        val_splits = []
        test_splits = []
        
        # Read split information from CSV
        with open(os.path.join(args.split_csv_root, f'splits_{fold}.csv'), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == '':
                    continue
                    
                # Extract split names
                train_name = row[1]
                val_name = row[2] if row[2] != '' else None
                test_name = row[3] if row[3] != '' else None
                
                # Match with labels and create split entries
                with open(args.label_csv, 'r') as f:
                    reader_label = csv.reader(f)
                    for row_label in reader_label:
                        if row_label[2] == train_name:
                            train_splits.append([train_name, row_label[-1], 
                                               os.path.join(args.prefix, f'{train_name}.pt')])
                        if val_name and row_label[2] == val_name:
                            val_splits.append([val_name, row_label[-1], 
                                             os.path.join(args.prefix, f'{val_name}.pt')])
                        if test_name and row_label[2] == test_name:
                            test_splits.append([test_name, row_label[-1], 
                                              os.path.join(args.prefix, f'{test_name}.pt')])
        
        # Save splits to pickle files
        pickle.dump(train_splits, open(os.path.join(args.root, f'train_splits_{fold}.pkl'), 'wb'))
        pickle.dump(val_splits, open(os.path.join(args.root, f'val_splits_{fold}.pkl'), 'wb'))
        pickle.dump(test_splits, open(os.path.join(args.root, f'test_splits_{fold}.pkl'), 'wb'))

if __name__ == '__main__':
    main()