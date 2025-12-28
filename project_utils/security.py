import os
import shutil
import tempfile

def validate_magic_number(header, extension):
    """
    Validates that the file header matches the expected magic number for the extension.

    Args:
        header (bytes): The first 256 bytes of the file.
        extension (str): The file extension (including dot), e.g., '.mp4'.

    Returns:
        bool: True if valid, False otherwise.
    """
    ext = extension.lower()

    # Signatures
    # MP4/MOV: ftyp at offset 4
    if ext in ['.mp4', '.mov', '.m4v']:
        if len(header) < 12: return False
        # Check for 'ftyp' at offset 4
        if header[4:8] == b'ftyp':
            return True
        # Some older MOV files might have different headers, but ftyp is standard for modern
        # Let's also check for 'moov' atom at start if ftyp is missing (rare)
        # Or 'mdat' (very rare as first atom)
        # But for security, strict ftyp is safer.
        return False

    # AVI: RIFF at 0, AVI at 8
    elif ext == '.avi':
        if len(header) < 12: return False
        if header[0:4] == b'RIFF' and header[8:12] == b'AVI ':
            return True
        return False

    # MKV/WebM: EBML ID 1A 45 DF A3
    elif ext in ['.mkv', '.webm']:
        if len(header) < 4: return False
        if header[0:4] == b'\x1a\x45\xdf\xa3':
            return True
        return False

    # Allow unknown extensions if they were passed in allowed_extensions?
    # No, if we don't know the signature, we can't validate it.
    # But to prevent breaking other types if this function is reused:
    # return True for unknown types?
    # For this specific application, we only allow video files.
    return False

def save_uploaded_file_securely(uploaded_file, max_size_mb=200, allowed_extensions=None):
    """
    Saves an uploaded file to a temporary file in a memory-efficient way (chunked write).
    Also performs basic size validation and extension checking.

    Args:
        uploaded_file: The file-like object from st.file_uploader.
        max_size_mb: Maximum allowed size in megabytes.
        allowed_extensions: Optional list of allowed file extensions (e.g., ['.mp4', '.mov']).
                          Case-insensitive. If None, all extensions are allowed.

    Returns:
        str: The path to the saved temporary file, or None if validation fails.
    """
    try:
        # Check size (Streamlit usually handles this, but defense in depth)
        # Note: uploaded_file.size might be available in some Streamlit versions/backends,
        # but reading in chunks allows us to stop if it exceeds limit during read.

        original_suffix = os.path.splitext(uploaded_file.name)[1].lower()

        # Defense in depth: Validate extension
        if allowed_extensions:
            # Normalize allowed extensions to ensure they have leading dot and are lowercase
            normalized_allowed = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_extensions]
            normalized_allowed = [ext.lower() for ext in normalized_allowed]

            if original_suffix not in normalized_allowed:
                raise ValueError(f"Invalid file extension: {original_suffix}. Allowed: {allowed_extensions}")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)

        chunk_size = 1024 * 1024 # 1MB chunks
        total_read = 0
        max_bytes = max_size_mb * 1024 * 1024

        # Security: Read header first for Magic Number Validation
        header_size = 256
        header = uploaded_file.read(header_size)
        total_read += len(header)

        if len(header) == 0:
             # Empty file
             tfile.close()
             os.remove(tfile.name)
             raise ValueError("File is empty")

        # Validate Magic Number
        if not validate_magic_number(header, original_suffix):
            tfile.close()
            os.remove(tfile.name)
            raise ValueError(f"File content does not match extension {original_suffix} (Magic Number Mismatch)")

        # Write header
        tfile.write(header)

        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break

            total_read += len(chunk)
            if total_read > max_bytes:
                tfile.close()
                os.remove(tfile.name)
                raise ValueError(f"File size exceeds limit of {max_size_mb}MB")

            tfile.write(chunk)

        tfile.close()
        return tfile.name
    except Exception as e:
        print(f"Error saving file: {e}")
        # Clean up if partially written
        if 'tfile' in locals() and os.path.exists(tfile.name):
            try:
                os.remove(tfile.name)
            except:
                pass
        return None
