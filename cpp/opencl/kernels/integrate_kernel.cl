__kernel void integrate_kernel(
    __global const int* axon_list,          // Input axon values
    __global const int* weight_list,        // Flattened 2D weight matrix
    __global int* potentials,               // Array for neuron potentials
    const int neuron_index,                 // Index of the neuron being updated
    const int num_axons                     // Number of axons
) {
    // Calculate the global index for this thread
    int axon_index = get_global_id(0);

    // Ensure the thread ID is within bounds
    if (axon_index < num_axons) {
        // Compute integration value for the neuron and axon
        int integration_value = axon_list[axon_index] *
                                weight_list[neuron_index * num_axons + axon_index];

        // Atomically update the neuron's potential to avoid race conditions
        atomic_add(&potentials[neuron_index], integration_value);
    }
}
