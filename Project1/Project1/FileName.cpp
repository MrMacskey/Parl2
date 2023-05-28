#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <file1> <file2> ... <fileN>" << endl;
        MPI_Finalize();
        return 1;
    }

    int num_files = argc - 1;
    vector<string> filenames(num_files);
    for (int i = 0; i < num_files; i++) {
        filenames[i] = string(argv[i + 1]);
    }

    vector<vector<double>> data(num_files);
    for (int i = 0; i < num_files; i++) {
        ifstream infile(filenames[i]);
        double val;
        while (infile >> val) {
            data[i].push_back(val);
        }
        infile.close();
    }

    int num_data = data[rank].size();
    vector<double> local_data = data[rank];
    sort(local_data.begin(), local_data.end());

    vector<int> sendcounts(size);
    vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        sendcounts[i] = data[i].size();
        if (i > 0) {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    }

    vector<double> sorted_data(num_data * size);
    MPI_Gatherv(local_data.data(), num_data, MPI_DOUBLE, sorted_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_files; i++) {
            string outfile_name = filenames[i] + "_res";
            ofstream outfile(outfile_name);
            int start_index = displs[i];
            int end_index = start_index + sendcounts[i];
            for (int j = start_index; j < end_index; j++) {
                outfile << sorted_data[j] << endl;
            }
            outfile.close();
        }
    }

    MPI_Finalize();
    return 0;
}