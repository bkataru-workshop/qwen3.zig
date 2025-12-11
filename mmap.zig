const std = @import("std");
const builtin = @import("builtin");

pub const MmapError = error{
    FileOpenFailed,
    StatFailed,
    EmptyFile,
    MmapFailed,
    MunmapFailed,
    WindowsCreateFileFailed,
    WindowsGetFileSizeFailed,
    WindowsCreateSectionFailed,
    WindowsMapViewFailed,
    InvalidPath,
};

pub const MappedFile = struct {
    data: []align(std.heap.page_size_min) const u8,

    impl: PlatformImpl,

    const PlatformImpl = union(enum) {
        posix: PosixImpl,
        windows: WindowsImpl,
    };

    const PosixImpl = struct {
        fd: std.posix.fd_t,
    };

    const WindowsImpl = struct {
        file_handle: std.os.windows.HANDLE,
        section_handle: std.os.windows.HANDLE,
    };

    pub fn init(path: []const u8) MmapError!MappedFile {
        switch (builtin.os.tag) {
            .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly => {
                return initPosix(path);
            },
        }
    }

    fn initPosix(path: []const u8) MmapError!MappedFile {
        const file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch {
            return MmapError.FileOpenFailed;
        };
        const fd = file.handle;

        const stat = file.stat() catch {
            std.posix.close(fd);
            return MmapError.StatFailed;
        };

        const file_size = stat.size;

        if (file_size == 0) {
            std.posix.close(fd);
            return MmapError.EmptyFile;
        }

        const mapped = std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ,
            .{ .TYPE = .SHARED },
            fd,
            0,
        ) catch {
            std.posix.close(fd);
            return MmapError.MmapFailed;
        };

        return MappedFile{
            .data = mapped,
            .impl = .{ .posix = .{ .fd = fd } },
        };
    }

    fn initWindows(path: []const u8) MmapError!MappedFile {
        if (builtin.os.tag != .windows) {
            unreachable;
        }

        const windows = std.os.windows;

        const file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch {
            return MmapError.FileOpenFailed;
        };
        const file_handle = file.handle;
    }
};
