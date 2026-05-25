/*
 * NS-3 supplemental experiment for the learning-augmented vacation queueing paper.
 *
 * Scenario:
 *   A single UAV station sends target traffic and rule-driven background traffic
 *   to a roadside/AP node over an 802.11 link. Remote ID, telemetry, and C2
 *   heartbeat are modeled as periodic background flows that occupy the shared
 *   wireless medium. Video or event-triggered C2 is modeled as the target flow.
 *   The optional two_sta_contention topology moves the rule-driven background
 *   flows to a second STA so that the target flow is tested under multi-station
 *   contention rather than co-located source traffic.
 *
 * Output:
 *   A packet-level CSV with transmit and receive timestamps. Postprocess it
 *   with scripts/ns3/postprocess_ns3.py to obtain window-level delay-risk
 *   metrics for the paper.
 *
 * Intended use:
 *   Copy this file into <ns-3-root>/scratch/uav_vehicular_vacation.cc and run:
 *     python ns3 run "uav_vehicular_vacation --scenario=overall --seed=7 --out=packets.csv"
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UavVehicularVacationExperiment");

namespace
{

struct SentPacketInfo
{
    std::string scenario;
    std::string flowName;
    bool target{false};
    uint32_t packetBytes{0};
    double txTime{0.0};
};

struct ExperimentState
{
    std::ofstream out;
    std::map<std::pair<uint32_t, uint32_t>, SentPacketInfo> sent;
    uint32_t seed{0};
};

ExperimentState g_state;

struct ScenarioConfig
{
    double loadScale{1.0};
    double vacationScale{1.0};
    double driftLevel{0.0};
    double channelScale{1.0};
    double uavSpeed{0.25};
    double startDistance{18.0};
};

ScenarioConfig
MakeScenario(const std::string& scenario)
{
    ScenarioConfig cfg;
    if (scenario == "load_high")
    {
        cfg.loadScale = 14.0;
        cfg.uavSpeed = 0.35;
    }
    else if (scenario == "vacation_high")
    {
        cfg.vacationScale = 8.0;
    }
    else if (scenario == "drift_strong")
    {
        cfg.loadScale = 2.0;
        cfg.vacationScale = 3.0;
        cfg.driftLevel = 1.4;
        cfg.channelScale = 1.25;
        cfg.uavSpeed = 2.25;
        cfg.startDistance = 45.0;
    }
    else if (scenario == "traffic_mix_video_heavy")
    {
        cfg.loadScale = 1.15;
    }
    else if (scenario == "traffic_mix_c2_heavy")
    {
        cfg.loadScale = 0.9;
    }
    return cfg;
}

void
WriteHeader(ExperimentState& state)
{
    state.out << "scenario,seed,flow_id,flow_name,target,seq,packet_bytes,tx_time,rx_time,delay,received\n";
}

void
WriteLossRows()
{
    for (const auto& item : g_state.sent)
    {
        const auto& key = item.first;
        const auto& info = item.second;
        g_state.out << info.scenario << "," << g_state.seed << "," << key.first << "," << info.flowName << ","
                    << (info.target ? 1 : 0) << "," << key.second << "," << info.packetBytes << ","
                    << info.txTime << ",,,0\n";
    }
}

class UdpTraceSink : public Application
{
  public:
    void Setup(uint16_t port)
    {
        m_port = port;
    }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
        m_socket->SetRecvCallback(MakeCallback(&UdpTraceSink::HandleRead, this));
    }

    void StopApplication() override
    {
        if (m_socket)
        {
            m_socket->Close();
        }
    }

    void HandleRead(Ptr<Socket> socket)
    {
        Address from;
        Ptr<Packet> packet;
        while ((packet = socket->RecvFrom(from)))
        {
            if (packet->GetSize() < 8)
            {
                continue;
            }
            uint8_t buffer[8];
            packet->CopyData(buffer, 8);

            uint32_t flowId = 0;
            uint32_t seq = 0;
            std::memcpy(&flowId, buffer, 4);
            std::memcpy(&seq, buffer + 4, 4);

            auto key = std::make_pair(flowId, seq);
            auto it = g_state.sent.find(key);
            if (it == g_state.sent.end())
            {
                continue;
            }

            const auto info = it->second;
            double rxTime = Simulator::Now().GetSeconds();
            double delay = rxTime - info.txTime;
            g_state.out << info.scenario << "," << g_state.seed << "," << flowId << "," << info.flowName << ","
                        << (info.target ? 1 : 0) << "," << seq << "," << info.packetBytes << ","
                        << info.txTime << "," << rxTime << "," << delay << ",1\n";
            g_state.sent.erase(it);
        }
    }

    Ptr<Socket> m_socket;
    uint16_t m_port{0};
};

class FlowTrafficApp : public Application
{
  public:
    void Setup(Address peer,
               uint32_t flowId,
               std::string scenario,
               std::string flowName,
               bool target,
               uint32_t packetBytes,
               double startTime,
               double stopTime,
               double meanInterval,
               bool poisson,
               uint32_t burstCount,
               double burstSpacing,
               uint32_t seed)
    {
        m_peer = peer;
        m_flowId = flowId;
        m_scenario = std::move(scenario);
        m_flowName = std::move(flowName);
        m_target = target;
        m_packetBytes = std::max<uint32_t>(packetBytes, 16);
        m_startTime = startTime;
        m_stopTime = stopTime;
        m_meanInterval = meanInterval;
        m_poisson = poisson;
        m_burstCount = std::max<uint32_t>(burstCount, 1);
        m_burstSpacing = burstSpacing;
        m_rng.seed(seed + 1009 * flowId);
    }

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        Simulator::Schedule(Seconds(m_startTime), &FlowTrafficApp::SendBurst, this);
    }

    void StopApplication() override
    {
        if (m_sendEvent.IsPending())
        {
            Simulator::Cancel(m_sendEvent);
        }
        if (m_socket)
        {
            m_socket->Close();
        }
    }

    double NextInterval()
    {
        if (m_poisson)
        {
            std::exponential_distribution<double> dist(1.0 / std::max(1e-6, m_meanInterval));
            return std::max(1e-4, dist(m_rng));
        }
        return m_meanInterval;
    }

    void SendOne()
    {
        double now = Simulator::Now().GetSeconds();
        if (now > m_stopTime)
        {
            return;
        }

        uint32_t seq = m_seq++;
        uint32_t payloadSize = std::max<uint32_t>(m_packetBytes, 16);
        uint8_t header[8];
        std::memcpy(header, &m_flowId, 4);
        std::memcpy(header + 4, &seq, 4);
        Ptr<Packet> packet = Create<Packet>(header, 8);
        if (payloadSize > 8)
        {
            packet->AddAtEnd(Create<Packet>(payloadSize - 8));
        }

        SentPacketInfo info;
        info.scenario = m_scenario;
        info.flowName = m_flowName;
        info.target = m_target;
        info.packetBytes = payloadSize;
        info.txTime = now;
        g_state.sent[std::make_pair(m_flowId, seq)] = info;

        m_socket->Send(packet);
    }

    void SendBurst()
    {
        double now = Simulator::Now().GetSeconds();
        if (now > m_stopTime)
        {
            return;
        }

        for (uint32_t i = 0; i < m_burstCount; ++i)
        {
            Simulator::Schedule(Seconds(i * m_burstSpacing), &FlowTrafficApp::SendOne, this);
        }

        double next = NextInterval();
        m_sendEvent = Simulator::Schedule(Seconds(next), &FlowTrafficApp::SendBurst, this);
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    EventId m_sendEvent;
    uint32_t m_flowId{0};
    std::string m_scenario;
    std::string m_flowName;
    bool m_target{false};
    uint32_t m_packetBytes{0};
    double m_startTime{0.0};
    double m_stopTime{0.0};
    double m_meanInterval{1.0};
    bool m_poisson{false};
    uint32_t m_burstCount{1};
    double m_burstSpacing{0.001};
    uint32_t m_seq{0};
    std::mt19937 m_rng;
};

Ptr<FlowTrafficApp>
InstallFlow(Ptr<Node> node,
            Address peer,
            uint32_t flowId,
            const std::string& scenario,
            const std::string& flowName,
            bool target,
            uint32_t packetBytes,
            double startTime,
            double stopTime,
            double meanInterval,
            bool poisson,
            uint32_t burstCount,
            double burstSpacing,
            uint32_t seed)
{
    Ptr<FlowTrafficApp> app = CreateObject<FlowTrafficApp>();
    app->Setup(peer,
               flowId,
               scenario,
               flowName,
               target,
               packetBytes,
               startTime,
               stopTime,
               meanInterval,
               poisson,
               burstCount,
               burstSpacing,
               seed);
    node->AddApplication(app);
    app->SetStartTime(Seconds(0.0));
    app->SetStopTime(Seconds(stopTime + 1.0));
    return app;
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string scenario = "overall";
    std::string target = "video";
    std::string out = "ns3_packets.csv";
    uint32_t seed = 7;
    double duration = 120.0;
    double warmup = 5.0;
    bool enablePcap = false;
    std::string topology = "single_sta";

    CommandLine cmd;
    cmd.AddValue("scenario", "Scenario name: overall, load_high, vacation_high, drift_strong", scenario);
    cmd.AddValue("target", "Target flow: video or event_c2", target);
    cmd.AddValue("out", "Output packet-level CSV path", out);
    cmd.AddValue("seed", "Random seed", seed);
    cmd.AddValue("duration", "Simulation duration in seconds", duration);
    cmd.AddValue("warmup", "Warm-up interval before traffic starts", warmup);
    cmd.AddValue("enablePcap", "Enable Wi-Fi pcap traces", enablePcap);
    cmd.AddValue("topology", "Topology: single_sta or two_sta_contention", topology);
    cmd.Parse(argc, argv);

    bool twoStaContention = (topology == "two_sta_contention");
    if (topology != "single_sta" && !twoStaContention)
    {
        NS_FATAL_ERROR("Unknown topology: " << topology);
    }
    std::string outputScenario = twoStaContention ? (scenario + "_two_sta") : scenario;

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(seed);
    g_state.seed = seed;

    ScenarioConfig cfg = MakeScenario(scenario);

    g_state.out.open(out.c_str(), std::ios::out);
    if (!g_state.out.is_open())
    {
        NS_FATAL_ERROR("Unable to open output file: " << out);
    }
    WriteHeader(g_state);

    NodeContainer nodes;
    nodes.Create(twoStaContention ? 3 : 2);
    Ptr<Node> uavNode = nodes.Get(0);
    Ptr<Node> apNode = nodes.Get(1);
    Ptr<Node> backgroundNode = twoStaContention ? nodes.Get(2) : uavNode;

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    phy.Set("TxPowerStart", DoubleValue(23.0));
    phy.Set("TxPowerEnd", DoubleValue(23.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue("ErpOfdmRate6Mbps"),
                                 "ControlMode",
                                 StringValue("ErpOfdmRate6Mbps"));

    WifiMacHelper mac;
    Ssid ssid = Ssid("uav-vehicular-vacation");
    NodeContainer staNodes;
    staNodes.Add(uavNode);
    if (twoStaContention)
    {
        staNodes.Add(backgroundNode);
    }
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevice = wifi.Install(phy, mac, staNodes);
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, apNode);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(cfg.startDistance, 0.0, 40.0));
    positionAlloc->Add(Vector(0.0, 0.0, 5.0));
    if (twoStaContention)
    {
        positionAlloc->Add(Vector(8.0, 18.0, 2.0));
    }
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(nodes);

    Ptr<ConstantVelocityMobilityModel> uavMobility =
        uavNode->GetObject<ConstantVelocityMobilityModel>();
    uavMobility->SetVelocity(Vector(cfg.uavSpeed, 0.0, 0.0));
    Ptr<ConstantVelocityMobilityModel> apMobility =
        apNode->GetObject<ConstantVelocityMobilityModel>();
    apMobility->SetVelocity(Vector(0.0, 0.0, 0.0));
    if (twoStaContention)
    {
        Ptr<ConstantVelocityMobilityModel> backgroundMobility =
            backgroundNode->GetObject<ConstantVelocityMobilityModel>();
        backgroundMobility->SetVelocity(Vector(0.0, 0.0, 0.0));
    }

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.7.0.0", "255.255.255.0");
    Ipv4InterfaceContainer staIf = address.Assign(staDevice);
    Ipv4InterfaceContainer apIf = address.Assign(apDevice);

    uint16_t sinkPort = 5000;
    Ptr<UdpTraceSink> sink = CreateObject<UdpTraceSink>();
    sink->Setup(sinkPort);
    apNode->AddApplication(sink);
    sink->SetStartTime(Seconds(0.0));
    sink->SetStopTime(Seconds(duration + 2.0));

    Address peerAddress = InetSocketAddress(apIf.GetAddress(0), sinkPort);

    double targetInterval = (target == "event_c2") ? 0.18 : 0.075;
    uint32_t targetBytes = (target == "event_c2") ? 220 : 1200;
    targetInterval = targetInterval / std::max(0.1, cfg.loadScale);
    targetBytes = static_cast<uint32_t>(targetBytes * cfg.channelScale);

    InstallFlow(uavNode,
                peerAddress,
                1,
                outputScenario,
                target,
                true,
                targetBytes,
                warmup,
                duration,
                targetInterval,
                true,
                1,
                0.0,
                seed);

    uint32_t ridBurst = static_cast<uint32_t>(std::max(1.0, std::round(1.0 * cfg.vacationScale)));
    uint32_t telemetryBurst =
        static_cast<uint32_t>(std::max(1.0, std::round(1.0 * cfg.vacationScale)));
    uint32_t heartbeatBurst =
        static_cast<uint32_t>(std::max(1.0, std::round(2.0 * cfg.vacationScale)));

    InstallFlow(backgroundNode,
                peerAddress,
                10,
                outputScenario,
                "remote_id",
                false,
                static_cast<uint32_t>(360 * cfg.vacationScale),
                warmup + 0.05,
                duration,
                1.0,
                false,
                ridBurst,
                0.002,
                seed);
    InstallFlow(backgroundNode,
                peerAddress,
                11,
                outputScenario,
                "telemetry",
                false,
                static_cast<uint32_t>(300 * cfg.vacationScale),
                warmup + 0.12,
                duration,
                0.55,
                false,
                telemetryBurst,
                0.002,
                seed);
    InstallFlow(backgroundNode,
                peerAddress,
                12,
                outputScenario,
                "c2_heartbeat",
                false,
                static_cast<uint32_t>(180 * cfg.vacationScale),
                warmup + 0.18,
                duration,
                0.22,
                false,
                heartbeatBurst,
                0.001,
                seed);

    if (enablePcap)
    {
        phy.EnablePcap("uav-vehicular-vacation-sta", staDevice.Get(0));
        phy.EnablePcap("uav-vehicular-vacation-ap", apDevice.Get(0));
    }

    Simulator::Stop(Seconds(duration + 2.0));
    Simulator::Run();
    WriteLossRows();
    Simulator::Destroy();
    g_state.out.close();

    return 0;
}
