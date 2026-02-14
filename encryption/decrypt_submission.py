#!/usr/bin/env python3
"""
Decrypt Submission - SERVER-SIDE ONLY (GitHub Actions)
"""

import sys
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def decrypt_submission(enc_file, private_key_path):
    """Decrypt encrypted submission"""
    
    print("="*70)
    print("🔓 DECRYPTING SUBMISSION")
    print("="*70)
    
    # Read encrypted file
    with open(enc_file, 'rb') as f:
        encrypted_data = f.read()
    
    print(f"📄 File: {enc_file}")
    print(f"📊 Size: {len(encrypted_data)} bytes")
    
    # Load private key
    with open(private_key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )
    
    # Parse
    key_length = int.from_bytes(encrypted_data[:4], 'big')
    encrypted_key = encrypted_data[4:4+key_length]
    iv = encrypted_data[4+key_length:4+key_length+16]
    ciphertext = encrypted_data[4+key_length+16:]
    
    # Decrypt AES key
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Decrypt data
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    # Save
    output_csv = enc_file.replace('.enc', '_decrypted.csv')
    with open(output_csv, 'wb') as f:
        f.write(plaintext)
    
    print(f"✅ Decrypted: {output_csv}")
    print("="*70)
    
    return output_csv

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python decrypt_submission.py <file.enc> <private_key.pem>")
        sys.exit(1)
    
    decrypt_submission(sys.argv[1], sys.argv[2])