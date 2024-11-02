import * as input_data from './input.json' with {type: "json"}
import * as config_data from './config.json' with {type: "json"}
import * as fs from 'fs'

const {packets, output_bus, cores} = input_data.default;
const {
    num_neurons,
    num_axons,
    num_cores_x,
    num_cores_y
} = config_data.default;

//console.log(packets);
//console.log(num_cores_x);

/////////////////////////////////packets.txt
let packetFile = fs.createWriteStream('../packets.txt', {encoding: 'utf8'});

for(const packet of packets) {
    //console.log(packet)
    let packetBin = Array(num_axons * num_cores_x).fill(0);

    for(const spike of packet) {
        packetBin[spike.destination_axon + num_axons * spike.destination_core[0]] = 1;
    }

    packetBin.forEach (spikeBin => {
        packetFile.write(spikeBin + " ");
    })

    packetFile.write("\n");
}

packetFile.end();

/////////////////////////////////images.txt - currently undeterminable. Will be set to 1 for all images for now
let imageFile = fs.createWriteStream('../images.txt', {encoding: 'utf8'});

for(let i = 0; i < packets.length; i++) {
    imageFile.write("1\n");
}

imageFile.end();

////////////////////////////////csram.txt
let csramFile = fs.createWriteStream('../csram.txt', {encoding: 'utf8'});

for(const core of cores) {
    let connectionBin = [];
    for(let neuronIdx = 0; neuronIdx < core.connections.length; neuronIdx++) {
        for(let axonConnection = 0; axonConnection < core.axons.length; axonConnection++) {
            if(core.connections[neuronIdx][axonConnection] === 0) connectionBin.push(0);
            else {
                const axon_type = core.axons[axonConnection];
                connectionBin.push(core.neurons[neuronIdx].weights[axon_type]);
            }
        }
        //console.log(core.connections[neuronIdx].length)
        //console.log(core.neurons[neuronIdx].weights)
    }

    connectionBin.forEach(weight => {
        csramFile.write(weight + " ");
    });

    csramFile.write("\n");
}

csramFile.end();

//network_topo.txt
let networkTopoFile = fs.createWriteStream('../network_topo.txt', {encoding: 'utf8'});

for(const core of cores) {
    networkTopoFile.write(`${core.coordinates[0]} ${core.coordinates[1]} ${core.axons.length} ${core.neurons.length} ${core.neurons[0].destination_core_offset[0] + core.coordinates[0]}`);
    networkTopoFile.write("\n");
    //console.log(core.neurons[0].destination_core_offset[0] + core.coordinates[0]);
}

networkTopoFile.write(`${output_bus.coordinates[0]} ${output_bus.coordinates[1]} ${0} ${output_bus.num_outputs}`);

networkTopoFile.end();

let coreOutputFile = fs.createWriteStream('../core_output.txt', {encoding: 'utf8'});

for(const core of cores) {
    for(const neuron of core.neurons) {
        coreOutputFile.write(`${neuron.destination_axon} `);
    }
    coreOutputFile.write("\n");
}

coreOutputFile.end();

//config.txt
let configFile = fs.createWriteStream('../config.txt', {encoding: 'utf8'});

//configFile.write(num_neurons + "\n");
//configFile.write(num_axons + "\n");
configFile.write(num_cores_x + "\n");
configFile.write(num_cores_y + "\n");

configFile.end();