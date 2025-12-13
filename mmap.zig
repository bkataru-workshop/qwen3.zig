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
            .windows => {
                return initWindows(path);
            },
            else => @compileError("Unsupported OS for memory mapping. Supported: Linux, macOS, FreeBSD, NetBSD, OpenBSD, DragonflyBSD, Windows"),
        }
    }

    pub fn initZ(path: [*:0]const u8) MmapError!MappedFile {
        return init(std.mem.sliceTo(path, 0));
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

        const file_size = windows.GetFileSizeEx(file_handle) catch {
            windows.CloseHandle(file_handle);
            return MmapError.WindowsGetFileSizeFailed;
        };

        if (file_size == 0) {
            windows.CloseHandle(file_handle);
            return MmapError.EmptyFile;
        }

        var section_handle: windows.HANDLE = undefined;
        const create_section_rc = windows.ntdll.NtCreateSection(
            &section_handle,
            windows.STANDARD_RIGHTS_REQUIRED | windows.SECTION_QUERY | windows.SECTION_MAP_READ,
            null,
            null,
            windows.PAGE_READONLY,
            windows.SEC_COMMIT,
            file_handle,
        );

        if (create_section_rc != .SUCCESS) {
            windows.CloseHandle(file_handle);
            return MmapError.WindowsCreateSectionFailed;
        }

        var view_size: usize = 0;
        var base_ptr: usize = 0;
        const map_section_rc = windows.ntdll.NtMapViewOfSection(
            section_handle,
            windows.GetCurrentProcess(),
            @ptrCast(&base_ptr),
            null,
            0,
            null,
            &view_size,
            .ViewUnmap,
            0,
            windows.PAGE_READONLY,
        );

        if (map_section_rc != .SUCCESS) {
            windows.CloseHandle(section_handle);
            windows.CloseHandle(file_handle);
            return MmapError.WindowsMapViewFailed;
        }

        const aligned_ptr: [*]align(std.heap.page_size_min) const u8 = @ptrFromInt(base_ptr);

        return MappedFile{
            .data = aligned_ptr[0..@intCast(file_size)],
            .impl = .{ .windows = .{
                .file_handle = file_handle,
                .section_handle = section_handle,
            } },
        };
    }

    pub fn deinit(self: *MappedFile) void {
        switch (builtin.os.tag) {
            .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly => {
                const posix_impl = self.impl.posix;
                _ = std.posix.system.munmap(@ptrCast(@constCast(self.data.ptr)), self.data.len);
                std.posix.close(posix_impl.fd);
            },
            .windows => {
                const windows = std.os.windows;
                const windows_impl = self.impl.windows;
                _ = windows.ntdll.NtUnmapViewOfSection(
                    windows.GetCurrentProcess(),
                    @ptrFromInt(@intFromPtr(self.data.ptr)),
                );
                windows.CloseHandle(windows_impl.section_handle);
                windows.CloseHandle(windows_impl.file_handle);
            },
            else => @compileError("Unsupported OS for memory mapping"),
        }
    }

    pub fn slice(self: *const MappedFile) []const u8 {
        return self.data;
    }

    pub fn len(self: *const MappedFile) usize {
        return self.data.len;
    }

    pub fn isEmpty(self: *const MappedFile) bool {
        return self.data.len == 0;
    }

    pub fn subslice(self: *const MappedFile, start: usize, end: usize) ?[]const u8 {
        if (start > end or end > self.data.len) {
            return null;
        }
        return self.data[start..end];
    }

    pub fn readAt(self: *const MappedFile, offset: usize, count: usize) ?[]const u8 {
        if (offset + count > self.data.len) {
            return null;
        }
        return self.data[offset..][0..count];
    }
};
