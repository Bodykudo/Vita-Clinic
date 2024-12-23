import { Injectable, UnauthorizedException } from '@nestjs/common';
import { Role } from '@prisma/client';

import { PrismaService } from 'src/prisma.service';

import {
  processAppointmentsData,
  processInvoicesData,
  processPatientsData,
} from './utils';

import {
  AdminDashboardGeneralStatisticsDto,
  DashboardAppointmentsDataDto,
  DashboardDoctorsAppointmentsDataDto,
  DashboardDoctorsSexDataDto,
  DashboardInvoicseDataDto,
  DashboardMedicalInsightsDto,
  DashboardMedicalServicesInsightsDto,
  DashboardPatientsAgeSexDataDto,
  DoctorDashboardGeneralStatisticsDto,
  GetAppointmentsDataQuery,
  GetInvoicesDataQuery,
} from './dto/dashboards.dto';

@Injectable()
export class DashboardsService {
  constructor(private prisma: PrismaService) {}

  async getAdminGeneralStatistics(): Promise<AdminDashboardGeneralStatisticsDto> {
    const patientsCount = await this.prisma.user.count({
      where: {
        role: 'patient',
        isActive: true,
      },
    });

    const devicesCount = await this.prisma.device.count();

    const doctorsCount = await this.prisma.user.count({
      where: {
        role: 'doctor',
        isActive: true,
      },
    });

    const appointmentsCountByStatus = await this.prisma.appointment.groupBy({
      by: 'status',
      _count: true,
    });

    const appointmentsCount = {
      all: appointmentsCountByStatus.reduce((acc, item) => {
        acc += item._count;
        return acc;
      }, 0),
      completed:
        appointmentsCountByStatus.find((count) => count.status === 'completed')
          ?._count || 0,
      approved:
        appointmentsCountByStatus.find((count) => count.status === 'approved')
          ?._count || 0,
      pending:
        appointmentsCountByStatus.find((count) => count.status === 'pending')
          ?._count || 0,
      cancelled:
        appointmentsCountByStatus.find((count) => count.status === 'cancelled')
          ?._count || 0,
      rejected:
        appointmentsCountByStatus.find((count) => count.status === 'rejected')
          ?._count || 0,
    };

    return {
      patientsCount,
      doctorsCount,
      appointmentsCount,
      devicesCount,
    };
  }

  async getDoctorGeneralStatistics(
    userId: string,
  ): Promise<DoctorDashboardGeneralStatisticsDto> {
    const patientsCount = await this.prisma.user.count({
      where: {
        role: 'patient',
        isActive: true,
      },
    });

    const devicesCount = await this.prisma.device.count();

    const doctorAppointmentsCount = await this.prisma.appointment.count({
      where: {
        status: 'approved',
        doctorId: userId,
      },
    });

    return {
      patientsCount,
      appointmentsCount: doctorAppointmentsCount,
      devicesCount,
    };
  }

  async getInvoicesData({
    startDate = new Date().toISOString(),
    endDate = new Date(
      new Date().getTime() + 7 * 24 * 60 * 60 * 1000,
    ).toISOString(),
  }: GetInvoicesDataQuery): Promise<DashboardInvoicseDataDto[]> {
    const startDateObj = new Date(startDate);
    startDateObj.setHours(0, 0, 0, 0);
    const endDateObj = new Date(endDate);
    endDateObj.setHours(23, 59, 59, 999);

    const rawCompletedBillings = await this.prisma.billing.groupBy({
      by: ['date'],
      where: {
        OR: [
          {
            status: 'paid',
          },
          { status: 'insurance' },
        ],
        date: {
          gte: startDateObj,
          lte: endDateObj,
        },
      },
      _sum: {
        amount: true,
      },
      orderBy: {
        date: 'asc',
      },
    });

    const rawPendingBillings = await this.prisma.billing.groupBy({
      by: ['date'],
      where: {
        status: 'initial',
        date: {
          gte: startDateObj,
          lte: endDateObj,
        },
      },
      _sum: {
        amount: true,
      },
      orderBy: {
        date: 'asc',
      },
    });

    const rawCancelledBillings = await this.prisma.billing.groupBy({
      by: ['date'],
      where: {
        status: 'cancelled',
        date: {
          gte: startDateObj,
          lte: endDateObj,
        },
      },
      _sum: {
        amount: true,
      },
      orderBy: {
        date: 'asc',
      },
    });

    const completedBillings = {
      id: 'Completed',
      data: processInvoicesData(
        new Date(startDate),
        new Date(endDate),
        rawCompletedBillings,
      ),
    };

    const pendingBillings = {
      id: 'Pending',
      data: processInvoicesData(
        new Date(startDate),
        new Date(endDate),
        rawPendingBillings,
      ),
    };

    const cancelledBillings = {
      id: 'Cancelled',
      data: processInvoicesData(
        new Date(startDate),
        new Date(endDate),
        rawCancelledBillings,
      ),
    };

    return [completedBillings, pendingBillings, cancelledBillings];
  }

  async getAppointmentsData({
    year = new Date().getFullYear(),
    status = 'completed',
  }: GetAppointmentsDataQuery): Promise<DashboardAppointmentsDataDto[]> {
    const startDate = new Date(year, 0, 1);
    const endDate = new Date(year + 1, 0, 1);

    const rawData = await this.prisma.appointment.groupBy({
      by: ['date'],
      where: {
        status: status === 'all' ? undefined : status,
        date: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        date: 'asc',
      },
      _count: true,
    });

    return processAppointmentsData(rawData);
  }

  async getPatientsAgeSexData(): Promise<DashboardPatientsAgeSexDataDto[]> {
    const rawData = await this.prisma.user.findMany({
      where: {
        role: 'patient',
        isActive: true,
      },
      select: {
        birthDate: true,
        sex: true,
      },
    });

    return processPatientsData(rawData);
  }

  async getDoctorsSexData(): Promise<DashboardDoctorsSexDataDto[]> {
    const data = await this.prisma.user.groupBy({
      by: 'sex',
      where: {
        role: 'doctor',
        isActive: true,
      },
      _count: true,
    });

    return data.map((item) => ({
      id: item.sex,
      value: item._count,
    }));
  }

  async getDoctorsAppointmentsData({
    startDate = new Date().toISOString(),
    endDate = new Date(
      new Date().getTime() + 7 * 24 * 60 * 60 * 1000,
    ).toISOString(),
  }: GetInvoicesDataQuery): Promise<DashboardDoctorsAppointmentsDataDto[]> {
    const startDateObj = new Date(startDate);
    startDateObj.setHours(0, 0, 0, 0);
    const endDateObj = new Date(endDate);
    endDateObj.setHours(23, 59, 59, 999);

    const rawData = await this.prisma.user.findMany({
      where: {
        role: 'doctor',
      },
      select: {
        id: true,
        firstName: true,
        lastName: true,
        _count: {
          select: {
            doctorAppointments: {
              where: {
                status: 'completed',
                date: {
                  gte: startDateObj,
                  lte: endDateObj,
                },
              },
            },
          },
        },
      },
    });

    const data = rawData
      .map((doctor) => ({
        id: `Dr. ${doctor.firstName} ${doctor.lastName}`,
        label: `Dr. ${doctor.firstName} ${doctor.lastName}`,
        value: doctor._count.doctorAppointments,
      }))
      .sort((a, b) => b.value - a.value);

    const topDoctors = data.slice(0, 4);
    const others = data.slice(4).reduce((acc, doctor) => acc + doctor.value, 0);
    const result = [
      ...topDoctors,
      { id: 'Others', label: 'Others', value: others },
    ];

    return result;
  }

  async _getDiagnosesData() {
    const data = await this.prisma.diagnosis.findMany({
      select: {
        name: true,
        _count: {
          select: {
            patientDiagnosis: true,
          },
        },
      },
      orderBy: {
        patientDiagnosis: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.patientDiagnosis,
    }));
  }

  async _getMedicationsData() {
    const data = await this.prisma.medication.findMany({
      select: {
        name: true,
        _count: {
          select: {
            patientMedication: true,
            prescriptions: true,
          },
        },
      },
      orderBy: [
        {
          patientMedication: {
            _count: 'desc',
          },
        },
        {
          prescriptions: {
            _count: 'desc',
          },
        },
      ],
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.patientMedication + item._count.prescriptions,
    }));
  }

  async _getSurgeriesData() {
    const data = await this.prisma.surgery.findMany({
      select: {
        name: true,
        _count: {
          select: {
            patientSurgery: true,
          },
        },
      },
      orderBy: {
        patientSurgery: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.patientSurgery,
    }));
  }

  async _getAllergiesData() {
    const data = await this.prisma.allergy.findMany({
      select: {
        name: true,
        _count: {
          select: {
            patientAllergy: true,
          },
        },
      },
      orderBy: {
        patientAllergy: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.patientAllergy,
    }));
  }

  async getMedicalInsights(): Promise<DashboardMedicalInsightsDto> {
    const diagnoses = await this._getDiagnosesData();
    const medications = await this._getMedicationsData();
    const surgeries = await this._getSurgeriesData();
    const allergies = await this._getAllergiesData();

    return {
      diagnoses,
      medications,
      surgeries,
      allergies,
    };
  }

  async _getServicesData() {
    const data = await this.prisma.service.findMany({
      select: {
        name: true,
        _count: {
          select: {
            appointmentServices: true,
          },
        },
      },
      orderBy: {
        appointmentServices: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.appointmentServices,
    }));
  }

  async _getTherapiesData() {
    const data = await this.prisma.therapy.findMany({
      select: {
        name: true,
        _count: {
          select: {
            appointmentServices: true,
          },
        },
      },
      orderBy: {
        appointmentServices: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.appointmentServices,
    }));
  }

  async _getScansData() {
    const data = await this.prisma.modality.findMany({
      select: {
        name: true,
        _count: {
          select: {
            appointmentServices: true,
          },
        },
      },
      orderBy: {
        appointmentServices: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.appointmentServices,
    }));
  }

  async _getLaboratoryTestsData() {
    const data = await this.prisma.laboratoryTest.findMany({
      select: {
        name: true,
        _count: {
          select: {
            appointmentServices: true,
          },
        },
      },
      orderBy: {
        appointmentServices: {
          _count: 'desc',
        },
      },
      take: 5,
    });

    return data.map((item) => ({
      name: item.name,
      count: item._count.appointmentServices,
    }));
  }

  async getMedicalServicesInsights(): Promise<DashboardMedicalServicesInsightsDto> {
    const services = await this._getServicesData();
    const therapies = await this._getTherapiesData();
    const scans = await this._getScansData();
    const laboratoryTests = await this._getLaboratoryTestsData();

    return {
      services,
      therapies,
      scans,
      laboratoryTests,
    };
  }
}
