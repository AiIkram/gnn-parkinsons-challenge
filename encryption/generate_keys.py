#!/usr/bin/env python3
"""
Generate RSA Key Pair for Encrypted Submissions
"""

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import os

print('='*70)
print('🔐 GENERATING RSA KEY PAIR')
print('='*70)

# Generate private key (4096-bit for strong security)
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
    backend=default_backend()
)

# Generate public key from private key
public_key = private_key.public_key()

# Ensure directory exists
os.makedirs('encryption', exist_ok=True)

# Save private key (KEEP SECRET!)
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('encryption/private_key.pem', 'wb') as f:
    f.write(private_pem)

print('✅ Private key saved: encryption/private_key.pem')
print('   ⚠️  DO NOT COMMIT THIS FILE!')

# Save public key (share with participants)
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('encryption/public_key.pem', 'wb') as f:
    f.write(public_pem)

print('✅ Public key saved: encryption/public_key.pem')
print('   ✓ This file should be committed to repo')

# Update .gitignore
with open('.gitignore', 'a') as f:
    f.write('\n# Encryption - Keep private key secret\n')
    f.write('encryption/private_key.pem\n')

print('✅ Updated .gitignore')
print('='*70)
print('NEXT STEPS:')
print('='*70)
print('1. Copy content of encryption/private_key.pem')
print('2. Add to GitHub Secrets as PRIVATE_KEY')
print('3. Commit encryption/public_key.pem to repository')
print('='*70)