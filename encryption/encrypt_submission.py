#!/usr/bin/env python3
"""
Encrypt Submission - Participants use this
Usage: python encryption/encrypt_submission.py predictions.csv
"""

import sys
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def encrypt_submission(csv_file):
    """Encrypt CSV file using public key"""
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    public_key_path = os.path.join(script_dir, 'public_key.pem')
    
    # Output filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = os.path.join('submissions', 'encrypted')
    output_file = os.path.join(output_dir, f'{base_name}.enc')
    
    print("="*70)
    print("🔐 ENCRYPTING SUBMISSION")
    print("="*70)
    print(f"📄 Input:  {csv_file}")
    print(f"🔑 Key:    {public_key_path}")
    print(f"💾 Output: {output_file}")
    print("="*70)
    
    # Read CSV
    if not os.path.exists(csv_file):
        print(f"❌ Error: File not found: {csv_file}")
        sys.exit(1)
    
    with open(csv_file, 'rb') as f:
        plaintext = f.read()
    
    print(f"📊 Size: {len(plaintext)} bytes")
    
    # Load public key
    if not os.path.exists(public_key_path):
        print(f"❌ Error: Public key not found!")
        print(f"   Looking for: {public_key_path}")
        sys.exit(1)
    
    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )
    
    # Generate random keys
    aes_key = os.urandom(32)  # 256-bit
    iv = os.urandom(16)       # 128-bit
    
    # Encrypt data with AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    
    # Encrypt AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Combine
    encrypted_data = (
        len(encrypted_key).to_bytes(4, 'big') +
        encrypted_key +
        iv +
        ciphertext
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(encrypted_data)
    
    print(f"✅ Encrypted: {len(encrypted_data)} bytes")
    print("="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print("\n📤 NEXT STEPS:")
    print(f"1. git add {output_file}")
    print('2. git commit -m "Submission: [Your Team Name]"')
    print("3. git push origin your-branch")
    print("4. Create Pull Request on GitHub")
    print("5. Wait 2-5 minutes for results")
    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python encryption/encrypt_submission.py <predictions.csv>")
        print("\nExample:")
        print("  python encryption/encrypt_submission.py submissions/my_predictions.csv")
        sys.exit(1)
    
    encrypt_submission(sys.argv[1])