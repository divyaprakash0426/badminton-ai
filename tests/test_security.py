import unittest
import io
import os
import shutil
from project_utils.security import save_uploaded_file_securely

class TestSecurity(unittest.TestCase):
    def test_save_valid_mp4(self):
        # Create a dummy MP4 file (minimal valid header)
        # 00 00 00 18 ftyp mp42 ...
        content = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 100
        f = io.BytesIO(content)
        f.name = "test.mp4"
        f.size = len(content)

        path = save_uploaded_file_securely(f, allowed_extensions=['mp4'])
        self.assertIsNotNone(path)
        if path and os.path.exists(path):
            os.remove(path)

    def test_save_invalid_extension(self):
        content = b'some content'
        f = io.BytesIO(content)
        f.name = "test.exe"
        f.size = len(content)

        # Expect None because validation fails and exception is caught
        path = save_uploaded_file_securely(f, allowed_extensions=['mp4'])
        self.assertIsNone(path)

    def test_save_masquerading_file(self):
        # An EXE file renamed to .mp4
        # 'MZ' is magic for DOS/Windows PE
        content = b'MZ' + b'\x00' * 100
        f = io.BytesIO(content)
        f.name = "malicious.mp4"
        f.size = len(content)

        # Should detect mismatch between extension (.mp4) and content (MZ)
        path = save_uploaded_file_securely(f, allowed_extensions=['mp4'])

        # Clean up if it failed to block
        if path and os.path.exists(path):
            os.remove(path)

        self.assertIsNone(path, "Failed to block file with invalid magic number")

if __name__ == '__main__':
    unittest.main()
