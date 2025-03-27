# -*- coding: utf-8 -*-

import zlib
from math import exp

import torch
from reedsolo import RSCodec

# Reed-Solomon codec for error correction
rs = RSCodec(250)

def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}.
    
    Args:
        text (str): Text to convert
        
    Returns:
        list: List of bits (0 or 1)
    """
    return bytearray_to_bits(text_to_bytearray(text))

def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text.
    
    Args:
        bits (list): List of bits (0 or 1)
        
    Returns:
        str: Decoded text
    """
    return bytearray_to_text(bits_to_bytearray(bits))

def bytearray_to_bits(x):
    """Convert bytearray to a list of bits.
    
    Args:
        x (bytearray): Bytearray to convert
        
    Returns:
        list: List of bits (0 or 1)
    """
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result

def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray.
    
    Args:
        bits (list): List of bits (0 or 1)
        
    Returns:
        bytearray: Converted bytearray
    """
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))

    return bytearray(ints)

def text_to_bytearray(text):
    """Compress and add error correction.
    
    Args:
        text (str): Text to convert
        
    Returns:
        bytearray: Compressed and encoded bytearray
    """
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x

def bytearray_to_text(x):
    """Apply error correction and decompress.
    
    Args:
        x (bytearray): Bytearray to convert
        
    Returns:
        str: Decoded text, or False if decoding fails
    """
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except Exception:
        return False

def patient_data_to_text(name, age, patient_id):
    """Convert patient data to a standardized text format.
    
    Args:
        name (str): Patient name
        age (str): Patient age
        patient_id (str): Patient ID
        
    Returns:
        str: Formatted patient data text
    """
    return f"Name: {name}\nAge: {age}\nID: {patient_id}"

def text_to_patient_data(text):
    """Extract patient data from text format.
    
    Args:
        text (str): Formatted patient data text
        
    Returns:
        dict: Patient data as a dictionary
    """
    lines = text.strip().split('\n')
    patient_data = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            patient_data[key.strip()] = value.strip()
    
    return patient_data