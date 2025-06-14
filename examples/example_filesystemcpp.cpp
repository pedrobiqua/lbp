// C++ program to list all the files and folder in the given
// directory
#include <filesystem>
#include <iostream>

using namespace std;
using namespace std::filesystem;

int main()
{
    // Define the directory path to list files from
    path directorypath = "";

    // To check if the directory exists or not
    if (exists(directorypath)
        && is_directory(directorypath)) {
        // Loop through each item (file or subdirectory) in
        // the directory
        for (const auto& entry :
             directory_iterator(directorypath)) {
            // Output the path of the file or subdirectory
            cout << "File: " << entry.path() << endl;
        }
    }
    else {
        // Handle the case where the directory doesn't exist
        cerr << "Directory not found." << endl;
    }

    return 0;
}