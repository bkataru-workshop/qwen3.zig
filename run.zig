/// Inference for GGUF Qwen-3 models in pure Zig
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
    data: []align(std.heap.page_size_min) const u8, // memory mapped data pointer
    impl: PlatformImpl,

    const PlatformImpl = union(enum) {
        posix: PosixImpl,
        windows: WindowsImpl,
    };

    const PosixImpl = struct {
        fd: std.posix.fd_t, // file descriptor for memory mapping
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
            windows.GetCUrrentProcess(),
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

    // size of the checkpoint file in bytes
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

// ----------------------------------------------------------------------------
// Transformer model
const Config = struct {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size
    seq_len: usize, // max sequence length
    head_dim: usize, // attention dimension
};

const TransformerWeights = struct {
    // token embedding table
    token_embedding_table: []const f32, // (vocab_size, dim)
    // weights for rmsnorms in each layer
    rms_att_weight: []const f32, // (layer, dim)
    rms_ffn_weight: []const f32, // (layer, dim)
    // weights for matmuls
    wq: []const f32, // (layer, dim, n_heads * head_dim)
    wk: []const f32, // (layer, dim, n_kv_heads * head_dim)
    wv: []const f32, // (layer, dim, n_kv_heads * head_dim)
    wo: []const f32, // (layer, n_heads * head_dim, dim)
    wq_norm: []const f32, // (layer, head_dim)
    wk_norm: []const f32, // (layer, head_dim)
    // weights for ffn. w1 = up, w3 = gate, w2 = down
    w1: []const f32, // (layer, dim, hidden_dim)
    w2: []const f32, // (layer, hidden_dim, dim)
    w3: []const f32, // (layer, dim, hidden_dim)
    // final rmsnorm
    rms_final_weight: []const f32, // (dim,)
    // Same as token_embedding_table. GGUF has the final layer anyway
    wcls: []const f32,

    pub fn mmap(data: []const u8, config: Config, header_offset: usize) !TransformerWeights {
        const float_data = try bytesAsFloats(data[header_offset..]);
        var offset: usize = 0;

        const wcls = float_data[offset .. offset + config.vocab_size * config.dim];
        offset += config.vocab_size * config.dim;
        const rms_final_weight = float_data[offset .. offset + config.dim];
        offset += config.dim;
        const token_embedding_table = float_data[offset .. offset + config.vocab_size * config.dim];
        offset += config.vocab_size * config.dim;
        const wk = float_data[offset .. offset + config.dim * (config.n_kv_heads * config.head_dim)];
        offset += config.dim * (config.n_kv_heads * config.head_dim);
        const wk_norm = float_data[offset .. offset + config.head_dim];
        offset += config.head_dim;
        const rms_att_weight = float_data[offset .. offset + config.dim];
        offset += config.dim;
        const wo = float_data[offset .. offset + (config.n_heads * config.head_dim) * config.dim];
        offset += (config.n_heads * config.head_dim) * config.dim;
        const wq = float_data[offset .. offset + config.dim * (config.n_heads * config.head_dim)];
        offset += config.dim * (config.n_heads * config.head_dim);
        const wq_norm = float_data[offset .. offset + config.head_dim];
    }

    fn bytesAsFloats(data: []const u8) ![]const f32 {
        if (data.len % 4 != 0) return error.ByteSliceLengthNotMultipleOf4;
        if (@intFromPtr(data.ptr) % 4 != 0) return error.DataNot4ByteAligned;

        const ptr = @as([*]const f32, @ptrCast(@alignCast(data.ptr)));
        return ptr[0..(data.len / 4)];
    }
};

pub fn numCalloc(comptime T: type, allocator: std.mem.Allocator, n: usize) std.mem.Allocator.Error![]T {
    // constrain to numeric types
    // TODO: can extend to encompass SIMD types as well?
    switch (@typeInfo(T)) {
        .Int, .Float => {},
        else => @compileError("Type '" ++ @typeName(T) ++ "' must be an integer or float"),
    }

    const slice = try allocator.alloc(T, n);
    std.mem.set(T, slice, 0);
    return slice;
}

const RunState = struct {
    allocator: std.mem.Allocator,
    // current wave of activations
    x: []f32, // activation at current time stamp (dim,)
    xb: []f32, // buffer (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    xb3: []f32, // an additional buffer just for convenience (att_head_dim,)
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, // query (att_head_dim,)
    k: []f32, // key (dim,)
    v: []f32, // value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // kv cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    pub fn malloc(allocator: std.mem.Allocator, p: Config) !RunState {
        const att_head_dim = p.n_heads * p.head_dim;
        const kv_dim = p.n_kv_heads * p.head_dim; // 1024

        return .{
            .allocator = allocator,
            .x = try numCalloc(f32, allocator, p.dim),
            .xb = try numCalloc(f32, allocator, p.dim),
            .xb2 = try numCalloc(f32, allocator, p.dim),
            .xb3 = try numCalloc(f32, allocator, att_head_dim),
            .hb = try numCalloc(f32, allocator, p.hidden_dim),
            .hb2 = try numCalloc(f32, allocator, p.hidden_dim),
            .q = try numCalloc(f32, allocator, att_head_dim),
            .k = try numCalloc(f32, allocator, kv_dim),
            .v = try numCalloc(f32, allocator, kv_dim),
            .att = try numCalloc(f32, allocator, p.n_heads * p.seq_len),
            .logits = try numCalloc(f32, allocator, p.vocab_size),
            .key_cache = try numCalloc(f32, allocator, p.n_layers * p.seq_len * kv_dim),
            .value_cache = try numCalloc(f32, allocator, p.n_layers * p.seq_len * kv_dim),
        };
    }

    pub fn free(self: RunState) void {
        const allocator = self.allocator;

        allocator.free(self.x);
        allocator.free(self.xb);
        allocator.free(self.xb2);
        allocator.free(self.xb3);
        allocator.free(self.hb);
        allocator.free(self.hb2);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.att);
        allocator.free(self.logits);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
    }
};

const Transformer = struct {
    config: Config, // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState, // buffers for the "wave" of activations in the forward pass
    mmap: MappedFile, // cross-platform, memory mapped file abstraction

    // read GGUF
    pub fn read_checkpoint(
        checkpoint_path: []const u8,
        config: *Config,
    ) !Transformer {}
};

pub fn main() void {
    std.debug.print("hello", .{});
}
