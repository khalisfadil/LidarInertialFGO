#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <boost/asio.hpp>

namespace slam {
    class UDPSocket {
    public:
        UDPSocket(boost::asio::io_context& context, 
                    const std::string& host, 
                    uint16_t port, 
                    std::function<void(const std::vector<uint8_t>&)> callback,
                    uint32_t bufferSize = 65535)
            : socket_(context, boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(host), port)),
                callback_(std::move(callback)) {  // Use std::move to avoid unnecessary copying of the callback
                buffer_.resize(bufferSize);         // Allocate memory for the buffer
                startReceive();                     // Start listening for incoming packets
        }

        ~UDPSocket() {
            stop();
        }
        
        void startReceive() {
            socket_.async_receive_from(
                boost::asio::buffer(buffer_),    // Buffer for receiving data
                senderEndpoint_,                // Endpoint to store sender's information
                [this](boost::system::error_code ec, std::size_t bytesReceived) {
                    if (!ec && bytesReceived > 0) {
                        // Copy received data into a vector
                        std::vector<uint8_t> packetData(buffer_.begin(), buffer_.begin() + bytesReceived);

                        #ifdef DEBUG
                        // Print received packet in hex format (for debugging purposes)
                        std::cout << "Received packet (size: " << bytesReceived << "): ";
                        for (size_t i = 0; i < bytesReceived; ++i) {
                            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                                    << static_cast<int>(packetData[i]) << " ";
                        }
                        std::cout << std::dec << std::endl;
                        #endif

                        // Invoke the user-defined callback with the received packet data
                        if (callback_) {
                            callback_(packetData);
                        }
                    } else if (ec) {
                        // Log an error if the receive operation failed
                        std::cerr << "Receive error: " << ec.message() << std::endl;
                    }

                    // Continue listening for the next packet
                    startReceive();
                }
            );
        }

        void stop() {
            boost::system::error_code ec;
            socket_.close(ec); // Close the socket
            if (ec) {
                std::cerr << "Error closing socket: " << ec.message() << std::endl;
            }
        }

    private:
        boost::asio::ip::udp::socket socket_;
        boost::asio::ip::udp::endpoint senderEndpoint_;
        std::vector<uint8_t> buffer_;
        std::function<void(const std::vector<uint8_t>&)> callback_;
    };

}  // namespace slam