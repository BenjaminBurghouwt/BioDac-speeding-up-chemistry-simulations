def print_summary(results, numpy_result, numpy_result_energy, closest_result):
    largest_deviation_distance = max(results['distance_deviations'])
    largest_deviation_energy = max(results['energy_deviations'])

    average_time = sum(results['vqe_times']) / len(results['vqe_times'])
    average_ram_usage = sum(results['ram_usages']) / len(results['ram_usages'])
    average_cpu_usage = sum(results['cpu_usages']) / len(results['cpu_usages'])
    average_total_cpu_usage = sum(results['total_cpu_usages']) / len(results['total_cpu_usages'])

    print(f"Largest deviation from exact result (distance): {largest_deviation_distance}")
    print(f"Largest deviation from exact result (energy): {largest_deviation_energy}")
    print(f"Average computation time: {average_time} seconds")

    print("\nBest VQE result:")
    print("Inter atomic distance: ", closest_result['distance'], "Angstrom - Exact: ", numpy_result, "Angstrom")
    print("Energy: ", closest_result['energy'], "Hartree - Exact: ", numpy_result_energy, "Hartree")

    print("\nPerformance metrics:")
    print(f"Average RAM usage: {average_ram_usage} GB")
    print(f"Average python CPU usage: {average_cpu_usage} %")
    print(f"Average total CPU usage: {average_total_cpu_usage} %")

    print("\n --- ")
    print("Due to printing a progress bar and keeping track of the CPU usage, the computation time is slightly higher "
          "than it would be without these features.")