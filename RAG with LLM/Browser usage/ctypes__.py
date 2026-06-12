import os, ctypes, ctypes.wintypes, sqlite3, tempfile, shutil

src = r"C:\Users\faiza\AppData\Local\Google\Chrome\User Data\Profile 23\Network\Cookies"
dst = os.path.join(tempfile.mkdtemp(), "Cookies")

print(f"Source: {src}")
print(f"Source size: {os.path.getsize(src)}")
print(f"Dest: {dst}")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
GENERIC_READ         = 0x80000000
FILE_SHARE_READ      = 0x00000001
FILE_SHARE_WRITE     = 0x00000002
FILE_SHARE_DELETE    = 0x00000004
OPEN_EXISTING        = 3
FILE_FLAG_SEQUENTIAL = 0x08000000
INVALID_HANDLE       = ctypes.wintypes.HANDLE(-1).value

h = k32.CreateFileW(
    src, GENERIC_READ,
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    None, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL, None
)

print(f"Handle: {h}  (INVALID={INVALID_HANDLE})")
if h == INVALID_HANDLE:
    err = ctypes.get_last_error()
    print(f"CreateFileW FAILED — WinError {err}")
else:
    print("CreateFileW SUCCESS — reading...")
    buf  = ctypes.create_string_buffer(1 << 20)
    read = ctypes.wintypes.DWORD(0)
    total = 0
    with open(dst, "wb") as out:
        while True:
            ok = k32.ReadFile(h, buf, len(buf), ctypes.byref(read), None)
            if not ok or read.value == 0:
                break
            out.write(buf.raw[:read.value])
            total += read.value
    k32.CloseHandle(h)
    print(f"Read {total} bytes")
    dest_size = os.path.getsize(dst)
    print(f"Dest size: {dest_size}")

    if dest_size > 0:
        with open(dst,"rb") as f: hdr=f.read(16)
        print(f"Header: {hdr}")
        print(f"Valid SQLite: {hdr[:15]==b'SQLite format 3'}")
        conn = sqlite3.connect(dst)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        print(f"Tables: {[r[0] for r in cur.fetchall()]}")
        cur.execute("SELECT COUNT(*) FROM cookies")
        print(f"Cookie count: {cur.fetchone()[0]}")
        conn.close()
    else:
        print("ERROR: dest file is empty!")

shutil.rmtree(os.path.dirname(dst), ignore_errors=True)
input("\nPress Enter...")