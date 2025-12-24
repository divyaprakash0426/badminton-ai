import os
import shutil
import tempfile

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
