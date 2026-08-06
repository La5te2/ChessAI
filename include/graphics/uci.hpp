// Runs a standard UCI engine as a child process and exposes thread-safe live
// analysis snapshots to the graphical application.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gadidae::graphics {

/// One engine-defined setting discovered from an `option name ...` UCI line.
struct UciOption {
	std::string name;
	std::string type;
	std::string default_value;
	std::optional<std::int64_t> minimum;
	std::optional<std::int64_t> maximum;
	std::vector<std::string> choices;

	bool operator==(const UciOption &) const = default;
};

/// Persistent process and protocol settings for one UCI engine.
struct EngineConfig {
	std::filesystem::path path;
	std::string name;
	std::string arguments;
	std::string options = "{}";
	int movetime_ms = 3000;
	std::uint64_t node_limit = 0;
	std::vector<UciOption> discovered_options;
	std::vector<std::string> button_commands;

	bool operator==(const EngineConfig &other) const {
		return path == other.path && name == other.name &&
			   arguments == other.arguments && options == other.options &&
			   movetime_ms == other.movetime_ms &&
			   node_limit == other.node_limit &&
			   button_commands == other.button_commands;
	}
};

/// One MultiPV row reported by an engine for the current position.
struct AnalysisLine {
	int multipv = 1;
	int depth = 0;
	int seldepth = 0;
	std::uint64_t nodes = 0;
	std::uint64_t nps = 0;
	int elapsed_ms = 0;
	bool mate = false;
	int score = 0;
	std::vector<std::string> pv;
};

/// Immutable copy of the latest search information consumed by the UI thread.
struct AnalysisSnapshot {
	std::uint64_t generation = 0;
	bool searching = false;
	bool finished = false;
	std::string bestmove;
	std::string engine_name;
	std::string error;
	std::vector<AnalysisLine> lines;
};

/// Owns one UCI subprocess, its reader thread, and the current analysis state.
class UciEngine {
public:
	UciEngine();
	~UciEngine();
	UciEngine(const UciEngine &) = delete;
	UciEngine &operator=(const UciEngine &) = delete;

	/// Launches and initializes the configured UCI process.
	void start(const EngineConfig &config);

	/// Launches and initializes the UCI process without blocking the GUI thread.
	void start_async(const EngineConfig &config);

	/// Stops the current search and terminates the child process.
	void close();

	/// Starts an asynchronous search from UCI position arguments including move history.
	std::uint64_t analyse(const std::string &position, bool infinite = false);

	/// Requests termination of the current search while keeping the engine alive.
	void stop_search();

	/// Returns a stable copy of the most recently parsed engine output.
	AnalysisSnapshot snapshot() const;

	/// Reports whether initialization completed and the process is still usable.
	bool ready() const;

	/// Reports whether a background handshake or model load is in progress.
	bool starting() const;

	/// Returns the effective engine name from the UCI handshake or user override.
	std::string display_name() const;

	/// Returns all option definitions reported by the current UCI engine.
	std::vector<UciOption> option_definitions() const;

	/// Performs only the UCI handshake and returns the engine-defined option schema.
	static std::vector<UciOption> discover_options(const EngineConfig &config);

private:
	class Process;
	struct AnalysisRequest {
		std::uint64_t generation = 0;
		std::string position;
		bool infinite = false;
	};

	/// Sends one complete line to the child process.
	void send(const std::string &line);
	/// Performs process creation, UCI handshake, and option configuration.
	void initialize(const EngineConfig &config);
	/// Sends position and go commands for a request that owns the output stream.
	void send_analysis(const AnalysisRequest &request);

	/// Waits for a handshake marker while checking process and parser errors.
	void wait_for_flag(const std::atomic_bool &flag,
					   std::chrono::milliseconds timeout,
					   const char *description);

	/// Continuously receives and parses stdout from the UCI process.
	void reader_loop();

	/// Updates engine identity and option metadata during the UCI handshake.
	void parse_handshake_line(const std::string &line);

	/// Parses an info line into the current MultiPV snapshot.
	void parse_info_line(const std::string &line);

	/// Parses bestmove and marks the active search as complete.
	void parse_bestmove_line(const std::string &line);

	/// Applies persistent option values and queued one-shot button commands.
	void configure();

	std::unique_ptr<Process> process_;
	EngineConfig config_;
	mutable std::mutex mutex_;
	std::mutex send_mutex_;
	std::condition_variable condition_;
	std::thread startup_;
	std::thread reader_;
	std::atomic_bool closing_{false};
	std::atomic_bool starting_{false};
	std::atomic_bool uciok_{false};
	std::atomic_bool readyok_{false};
	std::atomic_bool process_ready_{false};
	std::uint64_t generation_ = 0;
	bool command_searching_ = false;
	bool discard_output_ = false;
	std::optional<AnalysisRequest> pending_analysis_;
	AnalysisSnapshot snapshot_;
	std::string reported_name_;
	std::unordered_map<std::string, std::string> option_names_;
	std::vector<UciOption> option_definitions_;
};

} // namespace gadidae::graphics
