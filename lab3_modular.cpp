// lab3_modular.cpp
#include <systemc>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ostream>

// -------------------- Вспомогательные функции: float <-> uint32 --------------------
static inline uint32_t pack_f32(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}
static inline float unpack_f32(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}
static inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

// -------------------- Карта адресов (адресация по словам) ------------------
enum : uint32_t {
    ADDR_INPUT  = 0x0000, // N_in слов
    ADDR_W1     = 0x1000, // N_hidden * N_in слов
    ADDR_B1     = 0x2000, // N_hidden слов
    ADDR_HIDDEN = 0x3000, // N_hidden слов
    ADDR_W2     = 0x4000, // N_out * N_hidden слов
    ADDR_B2     = 0x5000, // N_out слов
    ADDR_OUTPUT = 0x6000  // N_out слов (логиты или softmax)
};

// -------------------- Запрос/ответ шины -------------------------
struct BusReq {
    int      master_id;   // кто отправил
    bool     is_write;
    uint32_t addr;        // адрес слова
    uint32_t wdata;
};

struct BusResp {
    int      master_id;
    uint32_t rdata;
};

// -------------------- Сообщения задания и завершения -----------------------
enum TaskKind : int { TK_HIDDEN = 0, TK_OUTPUT = 1 };

struct Task {
    int pe_id;     // какой PE должен выполнить
    int kind;      // скрытый/выходной слой
    int index;     // индекс нейрона
};

struct DoneMsg {
    int kind;
    int index;
};

// -------------------- Операторы вывода в поток (FIX для sc_fifo::print) ----
static inline std::ostream& operator<<(std::ostream& os, const DoneMsg& d) {
    return os << "DoneMsg{kind=" << d.kind << ", index=" << d.index << "}";
}
static inline std::ostream& operator<<(std::ostream& os, const Task& t) {
    return os << "Task{pe_id=" << t.pe_id << ", kind=" << t.kind << ", index=" << t.index << "}";
}
static inline std::ostream& operator<<(std::ostream& os, const BusReq& r) {
    return os << "BusReq{mid=" << r.master_id << ", wr=" << r.is_write
              << ", addr=" << r.addr << ", wdata=" << r.wdata << "}";
}
static inline std::ostream& operator<<(std::ostream& os, const BusResp& r) {
    return os << "BusResp{mid=" << r.master_id << ", rdata=" << r.rdata << "}";
}

// -------------------- Интерфейс памяти (порты/экспорты) --------------
struct IMemory : public sc_core::sc_interface {
    virtual uint32_t read_word(uint32_t addr) = 0;
    virtual void write_word(uint32_t addr, uint32_t data) = 0;
};

// -------------------- Модуль общей памяти (Shared Memory) --------------------------
SC_MODULE(SharedMemory), public IMemory {
    sc_core::sc_export<IMemory> mem_export;
    std::vector<uint32_t> mem;

    SC_CTOR(SharedMemory)
        : mem_export("mem_export"), mem(1 << 16, 0) // 65536 слов
    {
        mem_export.bind(*this);
    }

    uint32_t read_word(uint32_t addr) override {
        if (addr >= mem.size()) return 0;
        return mem[addr];
    }

    void write_word(uint32_t addr, uint32_t data) override {
        if (addr >= mem.size()) return;
        mem[addr] = data;
    }
};

// -------------------- Модуль шинной матрицы (Bus Matrix) -----------------------------
SC_MODULE(BusMatrix) {
    sc_core::sc_port<IMemory> mem_p;

    sc_core::sc_vector< sc_core::sc_fifo_in<BusReq>  > req_i;
    sc_core::sc_vector< sc_core::sc_fifo_out<BusResp> > resp_o;

    int n_masters;
    int rr_ptr;

    SC_HAS_PROCESS(BusMatrix);

    BusMatrix(sc_core::sc_module_name nm, int masters)
        : sc_core::sc_module(nm),
          req_i("req_i", masters),
          resp_o("resp_o", masters),
          n_masters(masters),
          rr_ptr(0)
    {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            bool did = false;

            for (int k = 0; k < n_masters; k++) {
                int i = (rr_ptr + k) % n_masters;
                if (req_i[i].num_available() > 0) {
                    BusReq rq = req_i[i].read();

                    // Время продвигается ТОЛЬКО здесь (коммуникация)
                    wait(1, sc_core::SC_NS);

                    BusResp rp;
                    rp.master_id = rq.master_id;

                    if (rq.is_write) {
                        mem_p->write_word(rq.addr, rq.wdata);
                        rp.rdata = 0;
                    } else {
                        rp.rdata = mem_p->read_word(rq.addr);
                    }

                    resp_o[i].write(rp);

                    rr_ptr = (i + 1) % n_masters;
                    did = true;
                    break;
                }
            }

            if (!did) {
                wait(1, sc_core::SC_NS);
            }
        }
    }
};

// -------------------- Модуль планировщика (Scheduler) ------------------------------
SC_MODULE(Scheduler) {
    sc_core::sc_fifo_in<Task> task_in;
    sc_core::sc_vector< sc_core::sc_fifo_out<Task> > task_to_pe;

    int n_pe;

    SC_HAS_PROCESS(Scheduler);

    Scheduler(sc_core::sc_module_name nm, int pe_count)
        : sc_core::sc_module(nm),
          task_in("task_in"),
          task_to_pe("task_to_pe", pe_count),
          n_pe(pe_count)
    {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            Task t = task_in.read();
            int id = t.pe_id % n_pe;
            task_to_pe[id].write(t);
        }
    }
};

// -------------------- Модуль вычислительного ядра (PE) -------------------------------------
SC_MODULE(PE) {
    int pe_id;

    sc_core::sc_fifo_out<BusReq>  req_o;
    sc_core::sc_fifo_in<BusResp>  resp_i;

    sc_core::sc_fifo_in<Task>     task_i;
    sc_core::sc_fifo_out<DoneMsg> done_o;

    int N_in, N_hidden, N_out;
    uint64_t mac_ops;

    SC_HAS_PROCESS(PE);

    PE(sc_core::sc_module_name nm, int id, int nin, int nh, int no)
        : sc_core::sc_module(nm),
          pe_id(id),
          req_o("req_o"), resp_i("resp_i"),
          task_i("task_i"), done_o("done_o"),
          N_in(nin), N_hidden(nh), N_out(no),
          mac_ops(0)
    {
        SC_THREAD(run);
    }

    uint32_t bus_read(uint32_t addr) {
        BusReq rq{pe_id, false, addr, 0};
        req_o.write(rq);
        BusResp rp = resp_i.read();
        return rp.rdata;
    }

    void bus_write(uint32_t addr, uint32_t data) {
        BusReq rq{pe_id, true, addr, data};
        req_o.write(rq);
        (void)resp_i.read();
    }

    void run() {
        while (true) {
            Task t = task_i.read();

            if (t.kind == TK_HIDDEN) {
                int j = t.index;
                float acc = unpack_f32(bus_read(ADDR_B1 + j));

                for (int i = 0; i < N_in; i++) {
                    float x  = unpack_f32(bus_read(ADDR_INPUT + i));
                    float w1 = unpack_f32(bus_read(ADDR_W1 + j * N_in + i));
                    acc += x * w1;
                    mac_ops++;
                }
                float h = relu(acc);
                bus_write(ADDR_HIDDEN + j, pack_f32(h));
            }
            else if (t.kind == TK_OUTPUT) {
                int k = t.index;
                float acc = unpack_f32(bus_read(ADDR_B2 + k));

                for (int j = 0; j < N_hidden; j++) {
                    float h  = unpack_f32(bus_read(ADDR_HIDDEN + j));
                    float w2 = unpack_f32(bus_read(ADDR_W2 + k * N_hidden + j));
                    acc += h * w2;
                    mac_ops++;
                }
                bus_write(ADDR_OUTPUT + k, pack_f32(acc));
            }

            done_o.write(DoneMsg{t.kind, t.index});
        }
    }
};

// -------------------- Модуль контроллера ввода-вывода (IO Controller) --------------------------
SC_MODULE(IOController) {
    sc_core::sc_fifo_out<BusReq> req_o;
    sc_core::sc_fifo_in<BusResp> resp_i;

    sc_core::sc_in<bool>  start_i;
    sc_core::sc_out<bool> done_o;

    int N_out;

    SC_HAS_PROCESS(IOController);

    IOController(sc_core::sc_module_name nm, int n_out)
        : sc_core::sc_module(nm),
          req_o("req_o"), resp_i("resp_i"),
          start_i("start_i"), done_o("done_o"),
          N_out(n_out)
    {
        SC_THREAD(run);
    }

    uint32_t bus_read(uint32_t addr) {
        BusReq rq{-1, false, addr, 0}; // мастер IO
        req_o.write(rq);
        BusResp rp = resp_i.read();
        return rp.rdata;
    }

    void run() {
        done_o.write(false);
        while (true) {
            while (!start_i.read()) wait(1, sc_core::SC_NS);

            std::cout << "\n[IO] Reading final outputs:\n";
            for (int k = 0; k < N_out; k++) {
                float y = unpack_f32(bus_read(ADDR_OUTPUT + k));
                std::cout << "  y[" << k << "] = " << y << "\n";
            }

            done_o.write(true);
            wait(1, sc_core::SC_NS);
            done_o.write(false);

            while (start_i.read()) wait(1, sc_core::SC_NS);
        }
    }
};

// -------------------- Модуль блока управления (Control Unit) ---------------------------
SC_MODULE(ControlUnit) {
    sc_core::sc_fifo_out<BusReq> req_o;
    sc_core::sc_fifo_in<BusResp> resp_i;

    sc_core::sc_fifo_out<Task> task_o;
    sc_core::sc_vector< sc_core::sc_fifo_in<DoneMsg> > done_i;

    sc_core::sc_out<bool> io_start_o;

    int N_PE;
    int N_in, N_hidden, N_out;

    SC_HAS_PROCESS(ControlUnit);

    ControlUnit(sc_core::sc_module_name nm, int n_pe, int nin, int nh, int no)
        : sc_core::sc_module(nm),
          req_o("req_o"), resp_i("resp_i"),
          task_o("task_o"),
          done_i("done_i", n_pe),
          io_start_o("io_start_o"),
          N_PE(n_pe), N_in(nin), N_hidden(nh), N_out(no)
    {
        SC_THREAD(run);
    }

    uint32_t bus_read(uint32_t addr) {
        BusReq rq{-2, false, addr, 0}; // мастер CU
        req_o.write(rq);
        BusResp rp = resp_i.read();
        return rp.rdata;
    }

    void bus_write(uint32_t addr, uint32_t data) {
        BusReq rq{-2, true, addr, data};
        req_o.write(rq);
        (void)resp_i.read();
    }

    void init_memory() {
        for (int i = 0; i < N_in; i++) {
            float x = (i % 7) * 0.1f;
            bus_write(ADDR_INPUT + i, pack_f32(x));
        }
        for (int j = 0; j < N_hidden; j++) {
            float b = 0.01f * (j - (N_hidden/2));
            bus_write(ADDR_B1 + j, pack_f32(b));
            for (int i = 0; i < N_in; i++) {
                    float w = 0.02f * ((i + j) % 5 - 2);
                    bus_write(ADDR_W1 + j * N_in + i, pack_f32(w));
            }
        }
        for (int k = 0; k < N_out; k++) {
            float b = 0.01f * (k - (N_out/2));
            bus_write(ADDR_B2 + k, pack_f32(b));
            for (int j = 0; j < N_hidden; j++) {
                float w = 0.02f * ((j + k) % 5 - 2);
                bus_write(ADDR_W2 + k * N_hidden + j, pack_f32(w));
            }
        }
    }

    void wait_done_count(int kind, int count) {
        int got = 0;
        while (got < count) {
            for (int p = 0; p < N_PE; p++) {
                if (done_i[p].num_available() > 0) {
                    DoneMsg d = done_i[p].read();
                    if (d.kind == kind) got++;
                }
            }
            wait(1, sc_core::SC_NS);
        }
    }

    void softmax_in_controlunit() {
        std::vector<float> z(N_out);
        for (int k = 0; k < N_out; k++) {
            z[k] = unpack_f32(bus_read(ADDR_OUTPUT + k));
        }
        float m = z[0];
        for (int k = 1; k < N_out; k++) m = std::max(m, z[k]);

        float sum = 0.0f;
        for (int k = 0; k < N_out; k++) {
            z[k] = std::exp(z[k] - m);
            sum += z[k];
        }
        for (int k = 0; k < N_out; k++) {
            float y = z[k] / sum;
            bus_write(ADDR_OUTPUT + k, pack_f32(y));
        }
    }

    void run() {
        io_start_o.write(false);

        init_memory();

        for (int j = 0; j < N_hidden; j++) {
            int pe = j % N_PE;
            task_o.write(Task{pe, TK_HIDDEN, j});
        }
        wait_done_count(TK_HIDDEN, N_hidden);

        for (int k = 0; k < N_out; k++) {
            int pe = k % N_PE;
            task_o.write(Task{pe, TK_OUTPUT, k});
        }
        wait_done_count(TK_OUTPUT, N_out);

        softmax_in_controlunit();

        io_start_o.write(true);
        wait(10, sc_core::SC_NS);
        io_start_o.write(false);

        wait(10, sc_core::SC_NS);
        sc_core::sc_stop();
    }
};

// -------------------- Верхнеуровневый модуль: NNProcessor -----------------------
SC_MODULE(NNProcessor) {
    int N_PE, N_in, N_hidden, N_out;

    SharedMemory mem;
    BusMatrix    bus;
    Scheduler    sched;

    std::vector<PE*> pes;

    IOController io;
    ControlUnit  cu;

    int n_masters;

    // FIXED: использовать creator-лямбды, а не sc_fifo(value)
    sc_core::sc_vector< sc_core::sc_fifo<BusReq> >   bus_req_fifo;
    sc_core::sc_vector< sc_core::sc_fifo<BusResp> >  bus_resp_fifo;

    sc_core::sc_fifo<Task> sched_in_fifo;
    sc_core::sc_vector< sc_core::sc_fifo<Task> >     pe_task_fifo;
    sc_core::sc_vector< sc_core::sc_fifo<DoneMsg> >  pe_done_fifo;

    sc_core::sc_signal<bool> io_start_sig;
    sc_core::sc_signal<bool> io_done_sig;

    SC_HAS_PROCESS(NNProcessor);

    NNProcessor(sc_core::sc_module_name nm, int n_pe, int nin, int nh, int no)
        : sc_core::sc_module(nm),
          N_PE(n_pe), N_in(nin), N_hidden(nh), N_out(no),
          mem("mem"),
          bus("bus", /*masters*/ (2 + n_pe)),
          sched("sched", n_pe),
          io("io", no),
          cu("cu", n_pe, nin, nh, no),
          n_masters(2 + n_pe),

          bus_req_fifo(
              "bus_req_fifo",
              n_masters,
              [](const char* name, sc_core::sc_vector_base::size_type) {
                  return new sc_core::sc_fifo<BusReq>(name, 64);
              }
          ),
          bus_resp_fifo(
              "bus_resp_fifo",
              n_masters,
              [](const char* name, sc_core::sc_vector_base::size_type) {
                  return new sc_core::sc_fifo<BusResp>(name, 64);
              }
          ),

          sched_in_fifo("sched_in_fifo", 128),

          pe_task_fifo(
              "pe_task_fifo",
              n_pe,
              [](const char* name, sc_core::sc_vector_base::size_type) {
                  return new sc_core::sc_fifo<Task>(name, 64);
              }
          ),
          pe_done_fifo(
              "pe_done_fifo",
              n_pe,
              [](const char* name, sc_core::sc_vector_base::size_type) {
                  return new sc_core::sc_fifo<DoneMsg>(name, 64);
              }
          )
    {
        // Привязать память к шине
        bus.mem_p(mem.mem_export);

        // Подключить FIFO шины
        for (int m = 0; m < n_masters; m++) {
            bus.req_i[m](bus_req_fifo[m]);
            bus.resp_o[m](bus_resp_fifo[m]);
        }

        // Индексы мастеров: 0 -> IO, 1 -> CU, 2.. -> PE
        io.req_o(bus_req_fifo[0]);
        io.resp_i(bus_resp_fifo[0]);

        cu.req_o(bus_req_fifo[1]);
        cu.resp_i(bus_resp_fifo[1]);

        // Подключения планировщика
        sched.task_in(sched_in_fifo);
        for (int p = 0; p < N_PE; p++) {
            sched.task_to_pe[p](pe_task_fifo[p]);
        }

        cu.task_o(sched_in_fifo);
        for (int p = 0; p < N_PE; p++) {
            cu.done_i[p](pe_done_fifo[p]);
        }

        // Управляющие сигналы IO
        io.start_i(io_start_sig);
        io.done_o(io_done_sig);
        cu.io_start_o(io_start_sig);

        // Создать и подключить PE
        pes.reserve(N_PE);
        for (int p = 0; p < N_PE; p++) {
            std::string name = "pe" + std::to_string(p);
            PE* pe = new PE(name.c_str(), /*id*/ (2 + p), N_in, N_hidden, N_out);

            pe->req_o(bus_req_fifo[2 + p]);
            pe->resp_i(bus_resp_fifo[2 + p]);

            pe->task_i(pe_task_fifo[p]);
            pe->done_o(pe_done_fifo[p]);

            pes.push_back(pe);
        }
    }

    ~NNProcessor() override {
        for (auto* pe : pes) delete pe;
    }
};

// -------------------- sc_main с параметром N_PE из командной строки -------------------------
int sc_main(int argc, char* argv[]) {
    int N_PE = 5;
    if (argc > 1) {
        int v = std::atoi(argv[1]);
        if (v > 0) N_PE = v;
    }

    // Выбор конфигурации сети (при необходимости можно изменить здесь)
    const int N_in = 49;
    const int N_hidden = 10; // 10 или 20
    const int N_out = 3;     // 3 или 5

    std::cout << "Running with N_PE=" << N_PE
              << " net=" << N_in << "-" << N_hidden << "-" << N_out << "\n";

    NNProcessor top("NNProcessor", N_PE, N_in, N_hidden, N_out);
    sc_core::sc_start();
    return 0;
}
