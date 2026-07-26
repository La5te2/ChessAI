// Implements the cross-platform child-process transport and incremental UCI
// parser used by both Simulator and Stadium.
#include "graphics/uci.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace gadidae::graphics {
namespace {

/// Returns an ASCII-lowercase copy for case-insensitive UCI option matching.
std::string lowercase(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
		return static_cast<char>(std::tolower(ch));
	});
	return value;
}

/// Removes leading and trailing ASCII whitespace from protocol text.
std::string trim(std::string value) {
	const auto first = value.find_first_not_of(" \t\r\n");
	if(first == std::string::npos) {
		return {};
	}
	const auto last = value.find_last_not_of(" \t\r\n");
	return value.substr(first, last - first + 1);
}

/// Splits a UCI line into whitespace-delimited protocol tokens.
std::vector<std::string> tokens_of(const std::string &line) {
	std::istringstream stream(line);
	std::vector<std::string> tokens;
	for(std::string token; stream >> token;) {
		tokens.push_back(std::move(token));
	}
	return tokens;
}

/// Builds a platform shell command while preserving the user-entered argument tail.
std::string command_text(const EngineConfig &config) {
	if(config.path.empty()) {
		throw std::invalid_argument("UCI engine path is empty");
	}
	std::string path = config.path.string();
	std::string command = "\"" + path + "\"";
	if(!trim(config.arguments).empty()) {
		command += " " + config.arguments;
	}
	return command;
}

#ifdef _WIN32
/// Converts UTF-8 command text into the UTF-16 representation required by Win32.
std::wstring widen(const std::string &value) {
	if(value.empty()) {
		return {};
	}
	const int length = MultiByteToWideChar(
		CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
	if(length <= 0) {
		throw std::runtime_error("could not convert command to UTF-16");
	}
	std::wstring result(static_cast<std::size_t>(length), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
					   result.data(), length);
	return result;
}
#endif

} // namespace

/// Bidirectional line transport around a platform child process.
class UciEngine::Process {
public:
	/// Starts the command with redirected standard input and merged stdout/stderr.
	explicit Process(const std::string &command) {
#ifdef _WIN32
		SECURITY_ATTRIBUTES security{};
		security.nLength = sizeof(security);
		security.bInheritHandle = TRUE;
		if(!CreatePipe(&stdout_read_, &stdout_write_child_, &security, 0) ||
		   !CreatePipe(&stdin_read_child_, &stdin_write_, &security, 0)) {
			throw std::runtime_error("could not create UCI pipes");
		}
		SetHandleInformation(stdout_read_, HANDLE_FLAG_INHERIT, 0);
		SetHandleInformation(stdin_write_, HANDLE_FLAG_INHERIT, 0);

		STARTUPINFOW startup{};
		startup.cb = sizeof(startup);
		startup.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
		startup.wShowWindow = SW_HIDE;
		startup.hStdInput = stdin_read_child_;
		startup.hStdOutput = stdout_write_child_;
		startup.hStdError = stdout_write_child_;
		std::wstring mutable_command = widen(command);
		mutable_command.push_back(L'\0');
		if(!CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
						   CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process_info_)) {
			const auto code = GetLastError();
			close_handles();
			throw std::runtime_error("could not start UCI engine, Win32 error " +
									 std::to_string(code));
		}
		CloseHandle(stdin_read_child_);
		stdin_read_child_ = nullptr;
		CloseHandle(stdout_write_child_);
		stdout_write_child_ = nullptr;
#else
		int input_pipe[2]{};
		int output_pipe[2]{};
		if(pipe(input_pipe) != 0 || pipe(output_pipe) != 0) {
			throw std::runtime_error("could not create UCI pipes: " +
									 std::string(std::strerror(errno)));
		}
		pid_ = fork();
		if(pid_ < 0) {
			throw std::runtime_error("could not fork UCI engine");
		}
		if(pid_ == 0) {
			dup2(input_pipe[0], STDIN_FILENO);
			dup2(output_pipe[1], STDOUT_FILENO);
			dup2(output_pipe[1], STDERR_FILENO);
			close(input_pipe[0]);
			close(input_pipe[1]);
			close(output_pipe[0]);
			close(output_pipe[1]);
			execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char *>(nullptr));
			_exit(127);
		}
		close(input_pipe[0]);
		close(output_pipe[1]);
		stdin_write_ = input_pipe[1];
		stdout_read_ = output_pipe[0];
#endif
	}

	/// Releases process handles after requesting a graceful, then forced, shutdown.
	~Process() {
		terminate();
	}

	/// Writes one newline-terminated UCI command atomically.
	void write_line(const std::string &line) {
		const std::string message = line + "\n";
#ifdef _WIN32
		DWORD written = 0;
		if(stdin_write_ == nullptr ||
		   !WriteFile(stdin_write_, message.data(), static_cast<DWORD>(message.size()),
					  &written, nullptr)) {
			throw std::runtime_error("failed to write to UCI engine");
		}
#else
		std::size_t offset = 0;
		while(offset < message.size()) {
			const auto count =
				::write(stdin_write_, message.data() + offset, message.size() - offset);
			if(count <= 0) {
				throw std::runtime_error("failed to write to UCI engine");
			}
			offset += static_cast<std::size_t>(count);
		}
#endif
	}

	/// Reads one complete line, returning false when the child closes stdout.
	bool read_line(std::string &line) {
		for(;;) {
			const auto newline = pending_.find('\n');
			if(newline != std::string::npos) {
				line = pending_.substr(0, newline);
				pending_.erase(0, newline + 1);
				if(!line.empty() && line.back() == '\r') {
					line.pop_back();
				}
				return true;
			}
			std::array<char, 4096> buffer{};
#ifdef _WIN32
			DWORD count = 0;
			if(stdout_read_ == nullptr ||
			   !ReadFile(stdout_read_, buffer.data(), static_cast<DWORD>(buffer.size()),
						 &count, nullptr) ||
			   count == 0) {
				return false;
			}
#else
			const auto count = ::read(stdout_read_, buffer.data(), buffer.size());
			if(count <= 0) {
				return false;
			}
#endif
			pending_.append(buffer.data(), static_cast<std::size_t>(count));
		}
	}

	/// Closes pipes and guarantees that the child process no longer survives.
	void terminate() {
		if(terminated_.exchange(true)) {
			return;
		}
		try {
			write_line("quit");
		} catch(...) {
		}
#ifdef _WIN32
		if(stdin_write_ != nullptr) {
			CloseHandle(stdin_write_);
			stdin_write_ = nullptr;
		}
		if(process_info_.hProcess != nullptr) {
			if(WaitForSingleObject(process_info_.hProcess, 800) == WAIT_TIMEOUT) {
				TerminateProcess(process_info_.hProcess, 0);
				WaitForSingleObject(process_info_.hProcess, 800);
			}
		}
		close_handles();
#else
		if(stdin_write_ >= 0) {
			close(stdin_write_);
			stdin_write_ = -1;
		}
		if(pid_ > 0) {
			for(int attempt = 0; attempt < 8; ++attempt) {
				int status = 0;
				if(waitpid(pid_, &status, WNOHANG) == pid_) {
					pid_ = -1;
					break;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			if(pid_ > 0) {
				kill(pid_, SIGTERM);
				waitpid(pid_, nullptr, 0);
				pid_ = -1;
			}
		}
		if(stdout_read_ >= 0) {
			close(stdout_read_);
			stdout_read_ = -1;
		}
#endif
	}

private:
#ifdef _WIN32
	/// Closes every Win32 handle that may have been created during startup.
	void close_handles() {
		for(HANDLE *handle : {&stdout_read_, &stdout_write_child_, &stdin_read_child_,
							  &stdin_write_, &process_info_.hThread,
							  &process_info_.hProcess}) {
			if(*handle != nullptr) {
				CloseHandle(*handle);
				*handle = nullptr;
			}
		}
	}

	HANDLE stdout_read_ = nullptr;
	HANDLE stdout_write_child_ = nullptr;
	HANDLE stdin_read_child_ = nullptr;
	HANDLE stdin_write_ = nullptr;
	PROCESS_INFORMATION process_info_{};
#else
	int stdin_write_ = -1;
	int stdout_read_ = -1;
	pid_t pid_ = -1;
#endif
	std::string pending_;
	std::atomic_bool terminated_{false};
};

UciEngine::UciEngine() = default;

UciEngine::~UciEngine() {
	close();
}

void UciEngine::start(const EngineConfig &config) {
	close();
	closing_ = false;
	starting_ = true;
	try {
		initialize(config);
		starting_ = false;
	} catch(...) {
		starting_ = false;
		throw;
	}
}

void UciEngine::start_async(const EngineConfig &config) {
	close();
	closing_ = false;
	starting_ = true;
	startup_ = std::thread([this, config] {
		try {
			initialize(config);
		} catch(const std::exception &error) {
			process_ready_ = false;
			if(!closing_) {
				std::lock_guard lock(mutex_);
				snapshot_.error = error.what();
				snapshot_.searching = false;
			}
		}
		starting_ = false;
		condition_.notify_all();
	});
}

void UciEngine::initialize(const EngineConfig &config) {
	config_ = config;
	uciok_ = false;
	readyok_ = false;
	process_ready_ = false;
	reported_name_.clear();
	option_names_.clear();
	{
		std::lock_guard lock(mutex_);
		snapshot_ = {};
		command_searching_ = false;
		discard_output_ = false;
		pending_analysis_.reset();
	}
	process_ = std::make_unique<Process>(command_text(config_));
	reader_ = std::thread(&UciEngine::reader_loop, this);
	send("uci");
	wait_for_flag(uciok_, std::chrono::seconds(15), "uciok");
	configure();
	readyok_ = false;
	send("isready");
	wait_for_flag(readyok_, std::chrono::seconds(15), "readyok");
	process_ready_ = true;
	{
		std::lock_guard lock(mutex_);
		snapshot_.engine_name = display_name();
	}
}

void UciEngine::close() {
	closing_ = true;
	process_ready_ = false;
	condition_.notify_all();
	if(startup_.joinable() &&
	   startup_.get_id() != std::this_thread::get_id()) {
		startup_.join();
	}
	{
		std::lock_guard lock(mutex_);
		pending_analysis_.reset();
		discard_output_ = true;
		snapshot_.searching = false;
	}
	if(process_) {
		try {
			send("stop");
			send("quit");
		} catch(...) {
		}
		process_->terminate();
	}
	if(reader_.joinable()) {
		reader_.join();
	}
	process_.reset();
	starting_ = false;
	{
		std::lock_guard lock(mutex_);
		snapshot_.searching = false;
		command_searching_ = false;
		discard_output_ = false;
		pending_analysis_.reset();
	}
}

std::uint64_t UciEngine::analyse(const std::string &fen, bool infinite) {
	if(!ready()) {
		throw std::runtime_error("UCI engine is not ready");
	}
	AnalysisRequest request;
	bool stop_previous = false;
	bool start_now = false;
	{
		std::lock_guard lock(mutex_);
		request = {++generation_, fen, infinite};
		snapshot_ = {};
		snapshot_.generation = request.generation;
		snapshot_.searching = true;
		snapshot_.engine_name = display_name();
		if(command_searching_) {
			pending_analysis_ = request;
			discard_output_ = true;
			stop_previous = true;
		} else {
			command_searching_ = true;
			discard_output_ = false;
			start_now = true;
		}
	}
	if(stop_previous) {
		send("stop");
	}
	if(start_now) {
		send_analysis(request);
	}
	return request.generation;
}

void UciEngine::send_analysis(const AnalysisRequest &request) {
	std::lock_guard send_lock(send_mutex_);
	if(!process_) {
		throw std::runtime_error("UCI process is not running");
	}
	process_->write_line("position fen " + request.fen);
	if(request.infinite) {
		process_->write_line("go infinite");
	} else {
		std::ostringstream go;
		go << "go";
		if(config_.node_limit > 0) {
			go << " nodes " << config_.node_limit;
		}
		if(config_.movetime_ms > 0 || config_.node_limit == 0) {
			go << " movetime " << std::max(0, config_.movetime_ms);
		}
		process_->write_line(go.str());
	}
}

void UciEngine::stop_search() {
	bool stop = false;
	{
		std::lock_guard lock(mutex_);
		pending_analysis_.reset();
		discard_output_ = true;
		snapshot_.searching = false;
		snapshot_.finished = false;
		stop = command_searching_;
	}
	if(stop && process_) {
		try {
			send("stop");
		} catch(...) {
		}
	}
}

AnalysisSnapshot UciEngine::snapshot() const {
	std::lock_guard lock(mutex_);
	return snapshot_;
}

bool UciEngine::ready() const {
	return process_ready_ && !closing_;
}

bool UciEngine::starting() const {
	return starting_ && !closing_;
}

std::string UciEngine::display_name() const {
	if(!trim(config_.name).empty()) {
		return trim(config_.name);
	}
	if(!reported_name_.empty()) {
		return reported_name_;
	}
	return config_.path.filename().string();
}

void UciEngine::send(const std::string &line) {
	std::lock_guard send_lock(send_mutex_);
	if(!process_) {
		throw std::runtime_error("UCI process is not running");
	}
	process_->write_line(line);
}

void UciEngine::wait_for_flag(const std::atomic_bool &flag,
							  std::chrono::milliseconds timeout,
							  const char *description) {
	std::unique_lock lock(mutex_);
	const bool completed = condition_.wait_for(lock, timeout, [&] {
		return flag.load() || !snapshot_.error.empty() || closing_.load();
	});
	if(!completed || !flag.load()) {
		const std::string detail = snapshot_.error.empty() ? "" : ": " + snapshot_.error;
		throw std::runtime_error(std::string("UCI handshake timed out waiting for ") +
								 description + detail);
	}
}

void UciEngine::reader_loop() {
	try {
		for(std::string line; !closing_ && process_ && process_->read_line(line);) {
			line = trim(std::move(line));
			if(line.empty()) {
				continue;
			}
			if(line == "uciok") {
				uciok_ = true;
				condition_.notify_all();
			} else if(line == "readyok") {
				readyok_ = true;
				condition_.notify_all();
			} else if(line.rfind("id ", 0) == 0 || line.rfind("option ", 0) == 0) {
				parse_handshake_line(line);
			} else if(line.rfind("info ", 0) == 0) {
				parse_info_line(line);
			} else if(line.rfind("bestmove ", 0) == 0) {
				parse_bestmove_line(line);
			}
		}
		if(!closing_) {
			std::lock_guard lock(mutex_);
			snapshot_.searching = false;
			snapshot_.error = "UCI engine closed its output stream";
			condition_.notify_all();
		}
	} catch(const std::exception &error) {
		std::lock_guard lock(mutex_);
		snapshot_.searching = false;
		snapshot_.error = error.what();
		condition_.notify_all();
	}
}

void UciEngine::parse_handshake_line(const std::string &line) {
	if(line.rfind("id name ", 0) == 0) {
		reported_name_ = trim(line.substr(8));
		return;
	}
	constexpr std::string_view prefix = "option name ";
	if(line.rfind(prefix, 0) != 0) {
		return;
	}
	const auto type = line.find(" type ", prefix.size());
	if(type == std::string::npos) {
		return;
	}
	const std::string name = trim(line.substr(prefix.size(), type - prefix.size()));
	option_names_[lowercase(name)] = name;
}

void UciEngine::parse_info_line(const std::string &line) {
	const auto tokens = tokens_of(line);
	AnalysisLine parsed;
	bool has_pv = false;
	for(std::size_t index = 1; index < tokens.size(); ++index) {
		const auto read_int = [&](int &target) {
			if(index + 1 < tokens.size()) {
				target = std::stoi(tokens[++index]);
			}
		};
		const auto read_uint = [&](std::uint64_t &target) {
			if(index + 1 < tokens.size()) {
				target = std::stoull(tokens[++index]);
			}
		};
		if(tokens[index] == "multipv") {
			read_int(parsed.multipv);
		} else if(tokens[index] == "depth") {
			read_int(parsed.depth);
		} else if(tokens[index] == "seldepth") {
			read_int(parsed.seldepth);
		} else if(tokens[index] == "nodes") {
			read_uint(parsed.nodes);
		} else if(tokens[index] == "nps") {
			read_uint(parsed.nps);
		} else if(tokens[index] == "time") {
			read_int(parsed.elapsed_ms);
		} else if(tokens[index] == "score" && index + 2 < tokens.size()) {
			parsed.mate = tokens[index + 1] == "mate";
			parsed.score = std::stoi(tokens[index + 2]);
			index += 2;
		} else if(tokens[index] == "pv") {
			parsed.pv.assign(tokens.begin() + static_cast<std::ptrdiff_t>(index + 1),
							 tokens.end());
			has_pv = !parsed.pv.empty();
			break;
		}
	}
	if(!has_pv) {
		return;
	}
	std::lock_guard lock(mutex_);
	if(discard_output_) {
		return;
	}
	auto &rows = snapshot_.lines;
	const auto existing = std::find_if(rows.begin(), rows.end(), [&](const auto &row) {
		return row.multipv == parsed.multipv;
	});
	if(existing == rows.end()) {
		rows.push_back(std::move(parsed));
	} else {
		*existing = std::move(parsed);
	}
	std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
		return left.multipv < right.multipv;
	});
}

void UciEngine::parse_bestmove_line(const std::string &line) {
	const auto tokens = tokens_of(line);
	std::optional<AnalysisRequest> next;
	{
		std::lock_guard lock(mutex_);
		command_searching_ = false;
		if(pending_analysis_) {
			next = std::move(pending_analysis_);
			pending_analysis_.reset();
			command_searching_ = true;
			discard_output_ = false;
		} else if(!discard_output_) {
			snapshot_.searching = false;
			snapshot_.finished = true;
			if(tokens.size() >= 2 && tokens[1] != "(none)") {
				snapshot_.bestmove = tokens[1];
			}
		}
		condition_.notify_all();
	}
	if(next) {
		send_analysis(*next);
	}
}

void UciEngine::configure() {
	nlohmann::json requested;
	try {
		requested = nlohmann::json::parse(
			trim(config_.options).empty() ? "{}" : config_.options);
	} catch(const std::exception &error) {
		throw std::invalid_argument("invalid UCI options JSON: " +
									std::string(error.what()));
	}
	if(!requested.is_object()) {
		throw std::invalid_argument("UCI options must be a JSON object");
	}
	const auto apply_option = [&](const std::string &requested_name,
								  const nlohmann::json &value) {
		const auto found = option_names_.find(lowercase(requested_name));
		if(found == option_names_.end()) {
			throw std::invalid_argument("UCI engine does not expose option " +
										requested_name);
		}
		std::string text;
		if(value.is_boolean()) {
			text = value.get<bool>() ? "true" : "false";
		} else if(value.is_string()) {
			text = value.get<std::string>();
		} else if(value.is_number()) {
			text = value.dump();
		} else {
			throw std::invalid_argument("unsupported UCI option value for " +
										requested_name);
		}
		send("setoption name " + found->second + " value " + text);
	};
	for(auto iterator = requested.begin(); iterator != requested.end(); ++iterator) {
		const auto key = lowercase(iterator.key());
		if(key == "device" || key == "multipv") {
			throw std::invalid_argument(
				iterator.key() +
				" is managed by its dedicated GUI field and must be removed from additional UCI options");
		}
		apply_option(iterator.key(), iterator.value());
	}
	if(lowercase(config_.device) != "auto") {
		const auto device = option_names_.find("device");
		if(device != option_names_.end()) {
			send("setoption name " + device->second + " value " + config_.device);
		}
	}
	const auto multipv = option_names_.find("multipv");
	if(multipv != option_names_.end()) {
		send("setoption name " + multipv->second + " value " +
			 std::to_string(std::max(1, config_.multipv)));
	}
}

} // namespace gadidae::graphics
