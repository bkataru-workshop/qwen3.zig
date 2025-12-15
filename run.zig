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
    allocator: std.mem.Allocator,
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size
    seq_len: usize, // max sequence length
    head_dim: usize, // attention dimension

    // load the GGUF config file
    pub fn load(allocator: std.mem.Allocator, filename: ?[]const u8) !Config {
        var file = try std.fs.cwd().openFile(filename orelse "header.txt", .{});
        defer file.close();

        var buf: [4096]u8 = undefined;
        var file_reader = file.reader(&buf);
        const r = &file_reader.interface;

        var config = std.StringHashMap(usize).init(allocator);
        defer config.deinit();

        while (r.takeDelimiterExclusive('\n')) |line| {
            // keep tossing the '\n' char until the end (where there isn't one)
            if (r.seek < r.end) r.toss(1);

            const cleaned = if (line.len != 0 and line[line.len - 1] == '\r')
                line[0 .. line.len - 1]
            else
                line;

            std.debug.print("line = {s}\n", .{cleaned});

            if (std.mem.indexOfScalar(u8, cleaned, '=') == null) continue;

            var parts = std.mem.splitScalar(u8, cleaned, '=');
            const key = parts.first();
            const value = parts.rest();

            if (key.len == 0 or value.len == 0) continue;

            if (std.mem.eql(u8, key, "QWEN3_EMBEDDING_LENGTH")) {
                try config.put("dim", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_FEED_FORWARD_LENGTH")) {
                try config.put("hidden_dim", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_BLOCK_COUNT")) {
                try config.put("n_layers", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_ATTENTION_HEAD_COUNT")) {
                try config.put("n_heads", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_ATTENTION_HEAD_COUNT_KV")) {
                try config.put("n_kv_heads", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_CONTEXT_LENGTH")) {
                try config.put("seq_len", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "QWEN3_ATTENTION_KEY_LENGTH")) {
                try config.put("head_dim", try std.fmt.parseInt(usize, value, 10));
            } else if (std.mem.eql(u8, key, "TOKENIZER_GGML_TOKENS")) {
                const ARRAY_LENGTH_KEY = "ARRAY_LENGTH=";

                if (std.mem.indexOf(u8, value, ARRAY_LENGTH_KEY)) |needle| {
                    const start = needle + ARRAY_LENGTH_KEY.len;
                    const subvalue = value[start..];
                    try config.put("vocab_size", try std.fmt.parseInt(usize, subvalue, 10));
                } else {
                    std.log.err("No key named '{s}' found in config", .{ARRAY_LENGTH_KEY});
                    return error.ConfigKeyNotFound;
                }
            }
        } else |err| switch (err) {
            error.EndOfStream => {},
            error.StreamTooLong => return err, // line didn't fit in buf
            else => return err,
        }

        if (config.count() != 8) {
            std.log.err("Invalid or corrupted config, didn't find exactly eight keys", .{});
            return error.InvalidOrCorruptedConfig;
        }

        return .{
            .allocator = allocator,
            .dim = config.get("dim").?,
            .hidden_dim = config.get("hidden_dim").?,
            .n_layers = config.get("n_layers").?,
            .n_heads = config.get("n_heads").?,
            .n_kv_heads = config.get("n_kv_heads").?,
            .seq_len = config.get("seq_len").?,
            .head_dim = config.get("head_dim").?,
            .vocab_size = config.get("vocab_size").?,
        };
    }
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
        offset += config.head_dim;
        const wv = float_data[offset .. offset + config.dim * (config.n_kv_heads * config.head_dim)];
        offset += config.dim * (config.n_kv_heads * config.head_dim);
        const w2 = float_data[offset .. offset + config.hidden_dim * config.dim];
        offset += config.hidden_dim * config.dim;
        const w3 = float_data[offset .. offset + config.dim * config.hidden_dim];
        offset += config.dim * config.hidden_dim;
        const rms_ffn_weight = float_data[offset .. offset + config.dim];
        offset += config.dim;
        const w1 = float_data[offset .. offset + config.dim * config.hidden_dim];
        offset += config.dim * config.hidden_dim;

        return .{
            .wcls = wcls,
            .rms_final_weight = rms_final_weight,
            .token_embedding_table = token_embedding_table,
            .wk = wk,
            .wk_norm = wk_norm,
            .rms_att_weight = rms_att_weight,
            .wo = wo,
            .wq = wq,
            .wq_norm = wq_norm,
            .wv = wv,
            .w2 = w2,
            .w3 = w3,
            .rms_ffn_weight = rms_ffn_weight,
            .w1 = w1,
        };
    }

    fn bytesAsFloats(data: []const u8) ![]const f32 {
        if (data.len % 4 != 0) return error.ByteSliceLengthNotMultipleOf4;
        if (@intFromPtr(data.ptr) % 4 != 0) return error.DataNot4ByteAligned;

        const ptr = @as([*]const f32, @ptrCast(@alignCast(data.ptr)));
        return ptr[0..(data.len / 4)];
    }
};

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

    fn numCalloc(comptime T: type, allocator: std.mem.Allocator, n: usize) std.mem.Allocator.Error![]T {
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

    pub fn malloc(allocator: std.mem.Allocator, config: Config) !RunState {
        const att_head_dim = config.n_heads * config.head_dim;
        const kv_dim = config.n_kv_heads * config.head_dim; // 1024

        return .{
            .allocator = allocator,
            .x = try numCalloc(f32, allocator, config.dim),
            .xb = try numCalloc(f32, allocator, config.dim),
            .xb2 = try numCalloc(f32, allocator, config.dim),
            .xb3 = try numCalloc(f32, allocator, att_head_dim),
            .hb = try numCalloc(f32, allocator, config.hidden_dim),
            .hb2 = try numCalloc(f32, allocator, config.hidden_dim),
            .q = try numCalloc(f32, allocator, att_head_dim),
            .k = try numCalloc(f32, allocator, kv_dim),
            .v = try numCalloc(f32, allocator, kv_dim),
            .att = try numCalloc(f32, allocator, config.n_heads * config.seq_len),
            .logits = try numCalloc(f32, allocator, config.vocab_size),
            .key_cache = try numCalloc(f32, allocator, config.n_layers * config.seq_len * kv_dim),
            .value_cache = try numCalloc(f32, allocator, config.n_layers * config.seq_len * kv_dim),
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
    allocator: std.mem.Allocator,
    config: Config, // the hyperparameters of the architecture (the blueprint)
    weights: TransformerWeights, // the weights of the model
    state: RunState, // buffers for the "wave" of activations in the forward pass
    mmap: MappedFile, // cross-platform, memory mapped file abstraction

    // read GGUF
    fn readCheckpoint(
        allocator: std.mem.Allocator,
        checkpoint_path: []const u8,
        config: Config,
    ) !Transformer {
        const mmap = try MappedFile.init(checkpoint_path);
        const header_offset = 5951648;

        std.log.info("file size is {d}", .{mmap.len()});

        const weights = try TransformerWeights.mmap(mmap.slice(), config, header_offset);
        const state = try RunState.malloc(allocator, config);

        return .{
            .allocator = allocator,
            .config = config,
            .weights = weights,
            .state = state,
            .mmap = mmap,
        };
    }

    pub fn build(allocator: std.mem.Allocator, checkpoint_path: []const u8, config: Config) !Transformer {
        return readCheckpoint(allocator, checkpoint_path, config) catch |err| {
            std.log.err("Error building Transformer: {}", .{err});
            return err;
        };
    }

    pub fn free(self: *Transformer) void {
        self.state.free();
        self.mmap.deinit(); // unmaps automatically
    }
};

pub fn main() void {
    std.debug.print("hello", .{});
}

test "configLoad" {
    const allocator = std.testing.allocator;

    const config = try Config.load(allocator, "header.txt");

    std.debug.print("config = {}", .{config});
}
