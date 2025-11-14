#include <systemc>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace sc_core;
using namespace sc_dt;
using namespace std;

struct NetConfig {
    unsigned input_size;
    unsigned hidden_size;
    unsigned output_size;
    const char* name;
};

SC_MODULE(NNProcessor) {
    const unsigned N_PE;                    // число вычислительных ядер
    vector<unsigned long long> pe_mac_ops;  // статистика MAC по ядрам

    SC_CTOR(NNProcessor)
    : N_PE(5), pe_mac_ops(5, 0)
    {
        SC_THREAD(run);
    }

    NNProcessor(sc_module_name name, unsigned n_pe)
    : sc_module(name), N_PE(n_pe), pe_mac_ops(n_pe, 0)
    {
        SC_THREAD(run);
    }

    // моделирование транзакции чтения из общей памяти через шину
    void bus_read(unsigned addr) {
        (void)addr;
        wait(1, SC_NS); // одна транзакция = 1 единица модельного времени
    }

    // моделирование транзакции записи в общую память через шину
    void bus_write(unsigned addr) {
        (void)addr;
        wait(1, SC_NS);
    }

    void reset_stats() {
        std::fill(pe_mac_ops.begin(), pe_mac_ops.end(), 0);
    }

    // моделирование одного скрытого нейрона на указанном PE
    void simulate_hidden_neuron(unsigned input_size, unsigned pe_id) {
        for (unsigned i = 0; i < input_size; ++i) {
            bus_read(i); // читаем вес
            bus_read(i); // читаем вход
            // операция MAC в нулевое модельное время
            pe_mac_ops[pe_id]++;
        }
        // ReLU: читаем и записываем результат
        bus_read(0);
        bus_write(0);
    }

    // моделирование одного выходного нейрона на указанном PE
    void simulate_output_neuron(unsigned hidden_size, unsigned pe_id) {
        for (unsigned i = 0; i < hidden_size; ++i) {
            bus_read(i); // вес
            bus_read(i); // активация скрытого слоя
            pe_mac_ops[pe_id]++;
        }
        bus_write(0); // запись линейного выхода
    }

    // моделирование Softmax на указанном PE (только транзакции памяти)
    void simulate_softmax(unsigned output_size, unsigned pe_id) {
        (void)pe_id;
        // читаем линейные выходы
        for (unsigned i = 0; i < output_size; ++i) {
            bus_read(i);
        }
        // записываем нормированные значения
        for (unsigned i = 0; i < output_size; ++i) {
            bus_write(i);
        }
        // exp, сумма и нормирование считаются вычислительными
        // операциями нулевой длительности
    }

    // моделирование расчёта сети заданной структуры
    void simulate_network(const NetConfig& cfg) {
        // скрытый слой: равномерное распределение нейронов по PE
        unsigned neurons_per_pe = (cfg.hidden_size + N_PE - 1) / N_PE;
        unsigned neuron_index = 0;

        for (unsigned pe = 0; pe < N_PE && neuron_index < cfg.hidden_size; ++pe) {
            unsigned assigned = std::min(neurons_per_pe, cfg.hidden_size - neuron_index);
            for (unsigned n = 0; n < assigned; ++n) {
                simulate_hidden_neuron(cfg.input_size, pe);
                neuron_index++;
            }
        }

        // выходной слой: используем не более 3 PE или N_PE
        unsigned used_out_pe = std::min<unsigned>(cfg.output_size, std::min<unsigned>(3, N_PE));
        for (unsigned o = 0; o < cfg.output_size; ++o) {
            unsigned pe = o % used_out_pe;
            simulate_output_neuron(cfg.hidden_size, pe);
        }

        // Softmax выполняем на двух последних ядрах (или одном, если N_PE < 2)
        if (N_PE >= 2) {
            simulate_softmax(cfg.output_size, N_PE - 2);
            simulate_softmax(cfg.output_size, N_PE - 1);
        } else {
            simulate_softmax(cfg.output_size, 0);
        }
    }

    void run() {
        NetConfig nets[3] = {
            {49, 10, 3, "Net1: 49-10-3"},
            {49, 20, 3, "Net2: 49-20-3"},
            {49, 10, 5, "Net3: 49-10-5"}
        };

        for (int k = 0; k < 3; ++k) {
            reset_stats();
            cout << "=== " << nets[k].name << " ===" << endl;
            sc_time t_start = sc_time_stamp();

            simulate_network(nets[k]);

            sc_time t_end = sc_time_stamp();
            sc_time elapsed = t_end - t_start;

            cout << "Model time: " << elapsed.to_double() << " ns" << std::endl;
            for (unsigned i = 0; i < N_PE; ++i) {
                cout << "PE" << i << " MAC ops: " << pe_mac_ops[i] << std::endl;
            }
            cout << "--------------------------" << endl;
        }

        sc_stop();
    }
};

int sc_main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;

    // 5 PE, как в ЛР2
    NNProcessor nnp("nnp", 5);
    sc_start();
    return 0;
}
